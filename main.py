import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import resnet
import pandas as pd
import os
import supcon_dataset
from torch.utils.tensorboard import SummaryWriter
from random import sample
import pandas as pd

from main_supcon import supcon_m
from main_linear import linear_m,show_confusion_matrix
import Metrics as metrics

def main():

    # 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    parser = argparse.ArgumentParser(description='Resnet18 CASME^2 Training')
    parser.add_argument('--outf', default='./model_checkpoints/', help='folder to output images and model checkpoints') #输出结果保存路径
    parser.add_argument('--datasets_test_csv', type=str, default='casme2/c3_test.csv',
                        help='casme2/c3_test.csv,casme2/c5_test.csv,'
                             'samm/samm_c3_test.csv,samm/samm_c5_test.csv'
                             'smic/smic_c3_test.csv,smic/smic_c5_test.csv')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    #LOSO
    df = pd.read_csv(
        os.path.join('/media/database/data4/hj/Code/flow_MER_constra_baseline/Resnet18/dataset', args.datasets_test_csv),
        sep=',', header=0)
    sub_id = df.iloc[:, 2].values #文件中的subid列
    subjects = list(set(sub_id)) #sub_id的集合
    sampleNum = len(subjects) #sub的个数
    mean_acc = 0
    mean_f1=0
    mean_uf1=0
    all_output = []
    all_labels = []

    with open("log.txt", "w") as f2:
        for id in subjects:
        #for id in range(19,20):
        #for id in [18,20]:#6,8,11,14,15,18,20
            # 训练
            if __name__ == "__main__":
                #supcon_train
                print("Leave Subject Name: %d,Start Training"%(id))
                loss = supcon_m(id)
                print('\nTraining Finished----------------------------------------')
                print("Train_LOSS=%.3f%%" % (loss))
                print('---------------------------------------------------------')

                #project head
                acc,f1,uf1,output,labels= linear_m(id)
                mean_acc += acc
                mean_f1 +=f1
                mean_uf1 +=uf1
                if acc !=0:
                   all_output.extend(output)
                   all_labels.extend(labels)

                matrix=show_confusion_matrix(all_output,all_labels)
                f2.write('LOSO_id: %03d | Acc: %.3f%% | F1: %.3f%%| UF1: %.3f%%'
                         % (id, acc, f1, uf1))
                f2.write('matrix:\n {}'.format(matrix))
                f2.write('\n')
                f2.flush()


        print("Finished--------------------------------------------------")
        ##评价指标
        pre=torch.tensor(all_output)
        lab=torch.tensor(all_labels)
        eval_acc = metrics.accuracy()
        eval_f1 = metrics.f1score()
        acc_w, acc_uw = eval_acc.eval(pre,lab)
        f1_w, f1_uw = eval_f1.eval(pre,lab)
        print('\nThe dataset has the ACC and F1:{:.4f} and {:.4f}'.format(acc_w, f1_w))
        print('\nThe dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))




if __name__ == '__main__':
    main()


