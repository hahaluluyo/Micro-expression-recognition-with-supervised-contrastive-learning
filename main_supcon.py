from __future__ import print_function

import os
import sys
import argparse
import time
import math
import supcon_dataset
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate,accuracy
from util import set_optimizer, save_model
from resnet import SupConResNet
from losses import SupConLoss
import interpretdl as it

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=2,
                        help='print frequency')#10
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')#50
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch_size')#256
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=70,#60
                        help='number of training epochs')#1000

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')#0.5
    parser.add_argument('--lr_decay_epochs', type=str, default='50,60',
                        help='where to decay lr, can be a list')#700,800,900
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='casme2_3',
                        choices=['casme2_5', 'casme2_3','samm_5', 'samm_3','smic_5','smic_3'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')#?
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')#0.07

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_model(opt):
    model = SupConResNet(name=opt.model) ##
    sd = model.resnet_base.state_dict()
    sd.update(torch.load('./resnet18.pth'))  # 加载预训练模型，通过torch.load(’.pth’) ,直接初始化新的神经网络对象
    model.resnet_base.load_state_dict(sd, strict=False)
    criterion= SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:#1
            model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    acc=AverageMeter()

    #end = time.time()
    for idx, data in enumerate(train_loader):

        images, labels, index=data
        #data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features= model(images)#tensor(40,128)
        data_batch=features

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)#(20,128)(20,128)  将tensor分成块结构,[,]为切分后的大小，dim为切分的维度
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)#(20,2,128)
        
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))


        # update metric
        losses.update(loss.item(), bsz)
        # compute acc
        # acc1, acc5 = accuracy(pre,labels, topk=(1, 5))
        # acc.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Tensorboard_feature_vis


        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: epoch：{0},batch_num：[{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  #'acc:{train_acc.val:.3f}'
                   .format(epoch, idx + 1, len(train_loader),loss=losses))
            sys.stdout.flush()

    return losses.avg

def supcon_m(sub_id):
    opt = parse_option()
    writer = SummaryWriter("con_log")

    # build data loader
    train_loader= supcon_dataset.getdata(sub_id)  # 读取并加载数据集

    # build model and criterion
    model, criterion= set_model(opt)#

    # build optimizer
    optimizer = set_optimizer(opt, model) #优化器：lr，使用动量(Momentum)的随机梯度下降法(SGD)，权重

    # training routine

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # tensorboard logger
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        #save_premodel
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)


    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    writer.close()




    return loss


if __name__ == '__main__':
    main()
    #supcon_m(1)
