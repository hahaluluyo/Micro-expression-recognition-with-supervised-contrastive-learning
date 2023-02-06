import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import torch
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--workers', type=int, default=0)#数据集较小时，不需要多个num_work
    parser.add_argument('--train_num', type=int, default=6)
    parser.add_argument('--test_num', type=int, default=11)
    parser.add_argument('--datasets_train_csv', type=str, default='casme2/c3_train.csv',#train中使用了双端帧策略
                        help='casme2/c3_train.csv,casme2/c5_train.csv,'
                             'samm/samm_c3_train.csv,samm/samm_c5_train.csv'
                             'smic/smic_c3_train.csv,smic/smic_c5_train.csv')

    parser.add_argument('--datasets_test_csv', type=str, default='casme2/c3_test.csv',#test中未使用双端帧策略
                        help='casme2/c3_test.csv,casme2/c5_test.csv,'
                             'samm/samm_c3_train.csv,samm/samm_c5_test.csv'
                             'smic/smic_c3_train.csv,smic/smic_c3_test.csv')
    return parser.parse_args()


'''
# 这样可以确定每次数据是怎么取的
def _init_fn(worker_id):
    np.random.seed(int(seed)+worker_id)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         pin_memory=True, worker_init_fn=_init_fn)
'''


class casme2DataSet(data.Dataset):
    def __init__(self, data_path, subid, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        # casme:0:Happiness, 1:Repression, 2:Surprise, 3:Disgust,4:Others
        # casme:0:Negative, 1:Positive,2:Surprise
        # samm: 0:Anger, 1:Contempt, 2:Happiness, 3:Surprise,4:Others
        # samm:0:Negative, 1:Positive,2:Surprise
        LABEL_COLUMN = 0
        NAME_COLUMN = 1
        SUB_COLUMN = 2
        args = parse_args()

        if args.datasets_test_csv == 'casme2/c5_test.csv' or args.datasets_test_csv == 'samm/samm_c5_test.csv':
            balance_class = 4  # 五分类 4=others
        else:
            balance_class = 0  # 三分类 0=negetive

        if phase == 'train':
            csv_path=args.datasets_train_csv
        else:
            csv_path =args.datasets_test_csv

        df = pd.read_csv(
            os.path.join('/media/database/data4/hj/Code/flow_MER_constra_baseline/Resnet18/dataset', csv_path),
            sep=',', header=0)

        file_names = df.iloc[:, NAME_COLUMN].values
        label = df.iloc[:, LABEL_COLUMN].values
        subject = df.iloc[:, SUB_COLUMN].values
        samplenum=len(subject)

        self.file_paths = []
        self.lab = []
        self.sub = []
        if phase == 'train':
            for j in range(0, len(data_path)):
                if j==4 or j==5 :#放大因子4,5仅用于平衡类别少的类# casme2&samm: j==4 or j==5  smic: j==5
                    for i in range(0,samplenum):
                        if subid!=subject[i] and label[i]!=balance_class:#leave one subject out,三分类balance去掉0，五分类balance去掉4
                            path = os.path.join(self.data_path[j], file_names[i])
                            self.file_paths.append(path)
                            self.lab.append(label[i])
                            self.sub.append(subject[i])
                else:
                    for i in range(0, samplenum):
                        if subid != subject[i]: #leave one subject out
                            path = os.path.join(self.data_path[j], file_names[i])
                            self.file_paths.append(path)
                            self.lab.append(label[i])
                            self.sub.append(subject[i])



        if phase == 'test':
            for j in range(0, len(data_path)):
                for i in range(0,samplenum):
                    if subid == subject[i]:
                        path = os.path.join(self.data_path[j], file_names[i])
                        self.lab.append(label[i])
                        self.file_paths.append(path)
                        self.sub.append(subject[i])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.copy()
        label = self.lab[idx]
        subject = self.sub[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label,subject


def getdata(subid):
    args = parse_args()

    # 加载train data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),  # resnet18对应的图片大小是224*224？
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    casme2_path_train = [
                   '/media/database/data4/hj/dataset/casM0',
                   '/media/database/data4/hj/dataset/casM1',
                   '/media/database/data4/hj/dataset/casM2',
                   '/media/database/data4/hj/dataset/casM3',
                   '/media/database/data4/hj/dataset/casM4',
                   '/media/database/data4/hj/dataset/casM5',
                   ]
    casme2_path_test = [
                    '/media/database/data4/hj/dataset/casM3',
                    ]

    samm_path_train = ['/media/database/data4/hj/dataset/SAMM/samM0',
                 '/media/database/data4/hj/dataset/SAMM/samM1',
                 '/media/database/data4/hj/dataset/SAMM/samM2',
                 '/media/database/data4/hj/dataset/SAMM/samM3',
                 '/media/database/data4/hj/dataset/SAMM/samM4',
                 '/media/database/data4/hj/dataset/SAMM/samM5',
                 ]
    samm_path_test = [
                       '/media/database/data4/hj/dataset/SAMM/samM3',
                       ]
    smic_path_train = ['/media/database/data4/hj/dataset/SMIC/smicM0',
                 '/media/database/data4/hj/dataset/SMIC/smicM1',
                 '/media/database/data4/hj/dataset/SMIC/smicM2',
                 '/media/database/data4/hj/dataset/SMIC/smicM3',
                 '/media/database/data4/hj/dataset/SMIC/smicM4',
                 '/media/database/data4/hj/dataset/SMIC/smicM5',
                 ]
    smic_path_test=[
        '/media/database/data4/hj/dataset/SMIC/smicM3',
    ]
    train_dataset = casme2DataSet(casme2_path_train, subid, phase='train', transform=data_transforms)
    print('Train set size:', train_dataset.__len__())
    # print('data',train_dataset)  #返回的是getitem中的内容，的是图片的tensor，lable，idx
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    # 加载test data

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    test_dataset=casme2DataSet(casme2_path_test, subid, phase='test',
                                          transform=data_transforms_val)
    test_loader=torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=args.batch_size,
                                                       num_workers=args.workers,
                                                       shuffle=False,
                                                       pin_memory=True)

    print('Test set size:',  test_dataset.__len__())

    return train_loader, test_loader
    # return train_dataset, test_dataset

