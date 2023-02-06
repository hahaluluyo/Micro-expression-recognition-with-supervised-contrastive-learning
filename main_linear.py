from __future__ import print_function

import sys
import argparse
import time
import math
import os
import torch
import torch.backends.cudnn as cudnn
import supcon_dataset
import linear_dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy,f1_score
from util import set_optimizer
from resnet import SupConResNet, LinearClassifier
from sklearn.metrics import confusion_matrix
from heatmap import plt_confusion_matrix
import Metrics as metrics
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=2,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=20,#20
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,#0.001
                        help='learning rate')#0.1
    parser.add_argument('--lr_decay_epochs', type=str, default='70',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset,
    parser.add_argument('--model',type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='casme2_3',
                        choices=['casme2_3', 'casme2_5','samm_3','samm_5',"smic_3","smic_5"], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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
    if opt.dataset == 'casme2_5' or opt.dataset == 'samm_5' or opt.dataset == 'smic_5':
        opt.n_cls = 5
    elif opt.dataset == 'casme2_3' or opt.dataset == 'samm_3' or opt.dataset == 'smic_3':
        opt.n_cls = 3
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.ckpt='./save/SupCon/{}_models/SupCon_{}_resnet18_lr_0.0001_decay_0.0001_bsz_20_temp_0.07_trial_0/last.pth'.format(opt.dataset,opt.dataset)

    return opt

def show_confusion_matrix(output,true,labels=[0,1,2]):#labels为类别c，c*c维矩阵 labels=[0,1,2,3,4]

    y_true = true
    y_pred=output
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt_confusion_matrix(matrix,normalize=False)
    return matrix
def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    print(opt.ckpt)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:#1
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)
    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt,writer):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, data in enumerate(train_loader):

        images, labels, index = data
        #print(torch.isnan(index))#判断是否有空值
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1,2))
        top1.update(acc1[0], bsz)

        # Adam  (SGDM)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()

    writer.add_scalar("train_loss", losses.avg, epoch)
    writer.add_scalar("train_acc", top1.avg, epoch)

    return losses.avg, top1.avg




def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    ##评价指标1
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    F1=AverageMeter()
    UF1=AverageMeter()
    all_output = []
    all_labels = []


    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            images, labels, index = data
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)
            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 2))
            top1.update(acc1[0], bsz)
            f1, uf1 = f1_score(output, labels)
            F1.update(f1 * 100, bsz)  # 小数化为百分制
            UF1.update(uf1 * 100, bsz)

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            output_v = torch.squeeze(pred, dim=0)

            all_output.extend(output_v.tolist())
            all_labels.extend(labels.tolist())

    return losses.avg,top1.avg,F1.avg,UF1.avg,all_output,all_labels


def linear_m(sub_id):

    opt = parse_option()
    writer=SummaryWriter('con_log')
    # build data loader
    train_loader, val_loader = linear_dataset.getdata(sub_id)
    # build model and criterion
    model, classifier, criterion = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    best_acc = 0
    best_f1 = 0
    best_uf1 = 0
    best_output = np.zeros((opt.n_cls,opt.n_cls),int).tolist()#创建n维0矩阵，转list
    best_labels = best_output

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        train_loss,train_acc=train(train_loader, model, classifier, criterion,optimizer, epoch, opt,writer)
        print('Subject {}, epoch {}, train_loss{:.2f}, train_acc {:.2f}'.format(sub_id,epoch, train_loss, train_acc))
        # eval
        loss, val_acc,f1,uf1,output,labels= validate(val_loader, model, classifier, criterion, opt)
        print('Subject {}, epoch {}, val_loss{:.2f}, val_acc {:.2f}'.format(sub_id,epoch, loss, val_acc))
        ##评价指标1
        writer.add_scalar("test_loss", loss, epoch)
        writer.add_scalar("test_acc", val_acc, epoch)
        writer.add_scalar("test_f1", f1, epoch)
        writer.add_scalar("test_uf1", uf1, epoch)



        if val_acc > best_acc:
            best_acc = val_acc
            best_output = output
            best_labels = labels
            best_f1 = f1
            best_uf1 = uf1


    print('\tSubject {} has the ACC:{:.4f} F1:{:.2f},UF1:{:.2f}\n'.format(sub_id, best_acc, best_f1, best_uf1))
    print('---------------------------\n')
    #show_confusion_matrix(output, labels)
    return best_acc,best_f1,best_uf1,best_output,best_labels


if __name__ == '__main__':
    main()
