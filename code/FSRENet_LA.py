import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.dataset import *
from networks.vnet import*
from datetime import datetime

from dataloaders import utils
from dataloaders.LA_dataset import (RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps, test_3d_patch
from val_3D import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='FSRENET_LA', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')  
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--labelnum', type=int,  default=4, help='trained samples')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=8,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40, help='consistency_rampup')

parser.add_argument('--rem', type=float, default=0.0925, help='')
parser.add_argument('--thresh', type=float, default=75, help='')
parser.add_argument('--max', type=float, default=95, help='')

parser.add_argument('--fsm', type=float, default=0.025, help='')
parser.add_argument('--balance', type=float, default=0.6, help='')


args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1) 
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def compute_unsupervised_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w, d = predict.shape
    Rdice_loss = losses.RDiceLoss(2)

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        # out = torch.unique(target,return_counts=True)
        # print(out)
        weight = batch_size * h * w *d/ torch.sum(target != 255)

    soft_predict =  torch.softmax(predict, dim=1)

    loss1 = weight * Rdice_loss(soft_predict, target.unsqueeze(1), ignore_index=255)  

    return loss1

patch_size = (112, 112, 80)

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2

    FSM_Ahasim = FSM_Ahasim_loss()
    FSM_Mixsim = FSM_Mixsim_loss()

    model1 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    model2 = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
  
    if args.rem:
        best_model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
        best_model.eval()

    model1.train()
    model2.train()

    db_train = LAHeart(base_dir=train_data_path,
                    split='train',
                    transform = transforms.Compose([
                        RandomRotFlip(),
                        RandomCrop(patch_size),
                        ToTensor(),
                        ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    best_performance1 = 0.0
    best_performance2 = 0.0
    performance1 = 0
    performance2 = 0
    performance_pre = 0.0
    iter_num = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    log_dir = os.path.join(snapshot_path, "logs",'{}_date_{}'.format(args.exp, datetime.now().strftime('%b%d_%H-%M-%S')))
    writer = SummaryWriter(log_dir)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):

            if args.rem:
                best_performance = (performance1 if (performance1 > performance2) else performance2)

                if best_performance > performance_pre and best_performance !=0 :
                    if performance1 > performance2:
                        best_model.load_state_dict(torch.load(save_mode_path1))
                        best_model.eval()
                        performance_pre = best_performance
                    else:
                        best_model.load_state_dict(torch.load(save_mode_path2))
                        best_model.eval()
                        performance_pre = best_performance

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1, o11, o12, o13, o14  = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2, o21, o22, o23, o24 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            if args.rem:
                outputs3, _,  _, _, _, = best_model(volume_batch[args.labeled_bs:])
                outputs_soft3 = torch.softmax(outputs3, dim=1)
                label_pred = torch.argmax(outputs_soft3, dim=1)
            
                #reliable_loss
            
                percent_unreliable = (100 - args.thresh) * (1 - iter_num / args.max_iterations)
                drop_percent = args.max - percent_unreliable
                reliable_loss1 = compute_unsupervised_loss(outputs1[args.labeled_bs:], label_pred, drop_percent, outputs3)
                reliable_loss2 = compute_unsupervised_loss(outputs2[args.labeled_bs:], label_pred, drop_percent, outputs3)
                reliable_loss = consistency_weight*(reliable_loss1 + reliable_loss2)

                writer.add_scalar('reliable_loss',args.rem*reliable_loss,iter_num)
                writer.add_scalar('reliable_loss1',reliable_loss1,iter_num)
                writer.add_scalar('reliable_loss2',reliable_loss2,iter_num)
            
            if args.fsm:

                intra_loss134, intra_cd134 = FSM_Ahasim(o12[args.labeled_bs:], o14[args.labeled_bs:], args.balance)

                intra_loss234, intra_cd234 = FSM_Ahasim(o22[args.labeled_bs:], o24[args.labeled_bs:], args.balance)

                intra_loss334, intra_cd334 = FSM_Mixsim(o12[args.labeled_bs:], o22[args.labeled_bs:], o14[args.labeled_bs:], o24[args.labeled_bs:], args.balance)
                
                writer.add_histogram('intra_cd134',intra_cd134,iter_num)
                writer.add_histogram('intra_cd234',intra_cd234,iter_num)
                writer.add_histogram('intra_cd334',intra_cd334,iter_num)
            
                intra_loss134 = intra_loss134.mean()
                intra_loss234 = intra_loss234.mean()
                intra_loss334 = intra_loss334.mean()

                intra_loss = intra_loss134 + intra_loss234 + intra_loss334 

                writer.add_scalar('intra_loss134', intra_loss134,iter_num)
                writer.add_scalar('intra_loss234', intra_loss234,iter_num)
                writer.add_scalar('intra_loss334', intra_loss334,iter_num)

     
                writer.add_scalar('intra_loss', args.fsm*intra_loss,iter_num)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            if args.fsm==0 and args.rem==0:
                loss = model1_loss + model2_loss
            elif args.fsm and args.rem==0:
                loss = model1_loss + model2_loss + args.fsm*intra_loss
            elif args.fsm==0 and args.rem:
                loss = model1_loss + model2_loss + args.rem*reliable_loss
            elif args.fsm and args.rem:
                loss = model1_loss + model2_loss + args.rem*reliable_loss + args.fsm*intra_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Model1_Predicted_label',
                                 grid_image, iter_num)

                image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Model2_Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 4 == 0 or iter_num==15000:
                model1.eval()
                
                performance1 = test_3d_patch.var_all_case_LA(model1, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if performance1 > best_performance1 or iter_num==15000:
                    best_performance1 = round(performance1, 4)
                    save_mode_path1 = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best_path1 = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path1)
                    torch.save(model1.state_dict(), save_best_path1)
                    logging.info("save best model to {}".format(save_mode_path1))
                writer.add_scalar('4_Var_dice/Dice', performance1, iter_num)
                writer.add_scalar('4_Var_dice/best_performance1', best_performance1, iter_num)
                model1.train()

                model2.eval()
                performance2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if performance2 > best_performance2 or iter_num==15000:
                    best_performance2= round(performance2, 4)
                    save_mode_path2 = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best_path2 = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path2)
                    torch.save(model2.state_dict(), save_best_path2)
                    logging.info("save best model to {}".format(save_mode_path2))
                writer.add_scalar('4_Var_dice/Dice', performance2, iter_num)
                writer.add_scalar('4_Var_dice/best_performance2', best_performance2, iter_num)
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
