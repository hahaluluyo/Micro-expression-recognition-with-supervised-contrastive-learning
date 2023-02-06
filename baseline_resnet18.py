from __future__ import print_function
import sys
import argparse
import time
import math
import os
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import linear_dataset
from torch.utils.tensorboard import SummaryWriter
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy,f1_score
from util import set_optimizer
from resnet import  ResNet,LinearClassifier
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
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=70,
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,#resnet18_lr=0.00001, classifier_lr=0.001
                        help='learning rate')#0.1
    parser.add_argument('--lr_decay_epochs', type=str, default='50,60',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='class_5',
                        choices=['class_5', 'class_3'], help='dataset')

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
    if opt.dataset == 'class_3':
        opt.n_cls = 3
    if opt.dataset == 'class_5':
        opt.n_cls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def show_confusion_matrix(output,true,labels=[0,1,2]):#labels为类别c，c*c维矩阵

    y_true = true
    y_pred=output
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    #plt_confusion_matrix(matrix,normalize=False)
    return matrix

def set_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(opt.model, 512).to(device)
    sd = model.resnet_base.state_dict()
    sd.update(torch.load('./resnet18.pth'))
    model.resnet_base.load_state_dict(sd, strict=False)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    if torch.cuda.is_available():

        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, classifier, criterion

def train_resnet(train_loader, model, opt,writer):

    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    # optimizer= optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99))
    optimizer = optim.SGD(model.parameters(),
                          lr=0.00001,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    mean_acc=0
    for epoch in range(50):#50
        model.train()

        correct = 0.0
        total = 0.0
        sum_loss=0.0
        for idx, data in enumerate(train_loader):
            images, labels, index = data
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
            optimizer.zero_grad()
            outputs=model(images)
            loss = criterion(outputs, labels)  # 交叉熵损失函数，求loss值
            loss.backward()#反向传播求梯度
            optimizer.step()#更新所有参数

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            loss=sum_loss / (idx + 1)
            acc=100. * correct / total
            print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, loss, acc))
            mean_acc=mean_acc+acc

            writer.add_scalar("resnet_train_loss", loss, epoch)
            writer.add_scalar("resnet_train_acc", acc, epoch)

    print('\nTraining Finished----------------------------------------')
    print("TotalEPOCH={} ,Best_Train_ACC={:.2f}".format( opt.epochs, (mean_acc/opt.epochs)))
    print('---------------------------------------------------------')
    return model

def train_linear(train_loader, model, classifier, criterion, optimizer, epoch, opt,writer):
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
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1,2))
        top1.update(acc1[0], bsz)

        # Adam /SGDM
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    writer.add_scalar("train_loss", losses.avg, epoch)
    writer.add_scalar("train_acc", top1.avg, epoch)

    return model,losses.avg, top1.avg


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
            output = classifier(model(images))
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


def main():



    opt = parse_option()
    # LOSO
    df = pd.read_csv(
        os.path.join('/media/database/data4/hj/Code/flow_MER_constra_baseline/Resnet18/dataset/casme2', 'c3_test.csv'),
        sep=',', header=0)
    sub_id = df.iloc[:, 2].values  # 文件中的subid列
    subjects = list(set(sub_id))  # sub_id的集合
    sampleNum = len(subjects)  # sub的个数
    # training routine

    all_output = []
    all_labels = []
    writer = SummaryWriter('bas_log')
    with open("base_log.txt", "w") as f:
        for id in subjects:
        #for id in range(17,18):
            # build data loader
            best_acc = 0
            best_f1 = 0
            best_uf1 = 0
            final_loss=0
            best_output = []
            best_labels = []
            train_loader, val_loader = linear_dataset.getdata(id)
            print("Leave Subject Name: %d,Start Training" % (id))

            model, classifier, criterion = set_model(opt)
            optimizer = set_optimizer(opt, classifier)
            #train_resnet
            model=train_resnet(train_loader,model,opt,writer)
            # train_linear
            for epoch in range(1, 11):
                adjust_learning_rate(opt, optimizer, epoch)
                model,train_loss,acc= train_linear(train_loader, model, classifier, criterion, optimizer, epoch,opt,writer)
                print('epoch {}, train_loss{:.2f}, train_acc {:.2f}'.format(epoch, train_loss, acc))
                # eval
                loss, val_acc, f1, uf1, output, labels = validate(val_loader, model, classifier, criterion, opt)
                print('epoch {}, test_loss{:.2f}, test_acc {:.2f}'.format(epoch, loss, val_acc))
                writer.add_scalar("test_loss", loss, epoch)
                writer.add_scalar("test_acc", val_acc, epoch)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_output = output
                    best_labels = labels
                    best_f1 = f1
                    best_uf1 = uf1
                final_loss=loss

            all_output.extend(best_output)
            all_labels.extend(best_labels)
            writer.add_scalar("subject_test_loss", final_loss, id)
            writer.add_scalar("subject_test_acc", best_acc, id)
            writer.add_scalar("subject_test_uf1", best_uf1, id)
            print('\tSubject {} has the ACC:{:.4f} F1:{:.2f},UF1:{:.2f}\n'.format(id, best_acc, best_f1, best_uf1))
            print('---------------------------\n')
            #matric
            matrix = show_confusion_matrix(all_output, all_labels)
            f.write('LOSO_id: %03d | Acc: %.3f%% | F1: %.3f%%| UF1: %.3f%%'
                     % (id, best_acc, best_f1, best_uf1))
            f.write('\n')
            f.write('matrix:\n {}'.format(matrix))
            f.write('\n')
            f.flush()


        ##评价指标2
        pre = torch.tensor(all_output)
        lab = torch.tensor(all_labels)
        eval_acc = metrics.accuracy()
        eval_f1 = metrics.f1score()
        acc_w, acc_uw = eval_acc.eval(pre, lab)
        f1_w, f1_uw = eval_f1.eval(pre, lab)
        print('\nThe dataset has the ACC and F1:{:.4f} and {:.4f}'.format(acc_w, f1_w))
        print('\nThe dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))
        all_matrix = show_confusion_matrix(all_output, all_labels)
        f.write('all_matrix:\n {}'.format(all_matrix))



if __name__ == '__main__':
    main()
