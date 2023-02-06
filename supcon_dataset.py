import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import torch
import os
import argparse
from util import TwoCropTransform, AverageMeter
from PIL import Image
from  torchvision import utils as vutils
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--workers', type=int, default=4)#数据集较小时，不需要多个num_work
    parser.add_argument('--datasets_csv',type=str,default='casme2/c3_train.csv',
                        help='casme2/c3_train.csv,casme2/c5_train.csv,'
                             'samm/samm_c3_train.csv,samm/samm_c3_train.csv'
                             'smic/smic_c3_train.csv')
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
        args = parse_args()
        LABEL_COLUMN = 0
        NAME_COLUMN = 1
        SUB_COLUMN = 2

        if  args.datasets_csv=='casme2/c5_train.csv' or args.datasets_csv=='samm/samm_c5_train.csv':
            balance_class=4 #五分类 4=others
        else :
            balance_class=0 # 三分类 0=negetive


        df = pd.read_csv(
            os.path.join('/media/database/data4/hj/Code/flow_MER_constra_baseline/Resnet18/dataset', args.datasets_csv),#'casme2/c3_train.csv'.'train.csv'
            sep=',', header=0)

        file_names = df.iloc[:, NAME_COLUMN].values
        label = df.iloc[:, LABEL_COLUMN].values
        # casme:0:Happiness, 1:Repression, 2:Surprise, 3:Disgust,4:Others
        # casme:0:Negative, 1:Positive,2:Surprise
        # samm: 0:Anger, 1:Contempt, 2:Happiness, 3:Surprise,4:Others
        # samm:0:Negative, 1:Positive,2:Surprise
        subject = df.iloc[:, SUB_COLUMN].values
        samplenum=len(subject)

        self.file_paths = []
        self.lab = []
        self.sub = []
        if phase == 'train':
            for j in range(0, len(data_path)):
                if j==4 or j==5:# 数据集为SMIC时，为平衡类间样本 j==0 or j==1 j==4 or j==5
                    for i in range(0, samplenum):
                        if subid != subject[i] and label[i]!= balance_class:# balance dataset
                            path = os.path.join(self.data_path[j], file_names[i])
                            self.file_paths.append(path)
                            self.lab.append(label[i])
                            self.sub.append(subject[i])

                else:
                    for i in range(0, samplenum):
                        if subid != subject[i]:
                            path = os.path.join(self.data_path[j], file_names[i])
                            self.file_paths.append(path)
                            self.lab.append(label[i])
                            self.sub.append(subject[i])


        if phase == 'test':
            for j in range(0,len(data_path)):
                for i in range(0,samplenum):
                    if subid == subject[i]:
                        path = os.path.join(self.data_path[j], file_names[i])
                        self.lab.append(label[i])
                        self.sub.append(subject[i])
                        self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            print(path)
        image = image[:, :, ::-1]  # BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.copy() # H W C
        label = self.lab[idx]
        subject = self.sub[idx]
        if self.transform is not None:
            image = Image.fromarray(image) #CWH
            image = self.transform(image)
        # 查看数据增强后的图像
        #if idx==8:
        # img_tensor = image[0] # CHW
        # plt.figure()
        # img = img_tensor.numpy().transpose((1, 2, 0)) #  HW
        # img = np.clip(img, 0, 1)
        # plt.imshow(img)
        # plt.show()
        # plt.pause(1)
        # plt.close()

        return image, label, subject


def getdata(subid):
    args = parse_args()

    # 加载train data
    data_transforms = transforms.Compose([
         #transforms.RandomResizedCrop(size=224, scale=(0.8, 1.)),#功能：随机长宽裁剪原始照片，最后将照片resize到设定好的size
         transforms.RandomHorizontalFlip(p=0.5),#水平翻转0.2
         transforms.RandomApply([
             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
         ], p=0.5),#0.2
         transforms.RandomGrayscale(p=0.2),#转灰度图
         # transforms.ToPILImage(),
         transforms.Resize((224,224)),#HW
         transforms.ToTensor(),
         # transforms.RandomErasing(p=0.2,scale=(0.02, 0.16),ratio =(0.5, 0.5),value = 0,inplace = False)
         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
         #                      std=[0.229, 0.224, 0.225])
     ])
    casme2_path=['/media/database/data4/hj/dataset/casM0',
                 '/media/database/data4/hj/dataset/casM1',
                 '/media/database/data4/hj/dataset/casM2',
                 '/media/database/data4/hj/dataset/casM3',
                 '/media/database/data4/hj/dataset/casM4',
                 '/media/database/data4/hj/dataset/casM5',
                 ]
    samm_path= [ '/media/database/data4/hj/dataset/SAMM/samM0',
                 '/media/database/data4/hj/dataset/SAMM/samM1',
                 '/media/database/data4/hj/dataset/SAMM/samM2',
                 '/media/database/data4/hj/dataset/SAMM/samM3',
                 '/media/database/data4/hj/dataset/SAMM/samM4',
                 '/media/database/data4/hj/dataset/SAMM/samM5',
                 ]
    smic_path = ['/media/database/data4/hj/dataset/SMIC/smicM0',
                 '/media/database/data4/hj/dataset/SMIC/smicM1',
                 '/media/database/data4/hj/dataset/SMIC/smicM2',
                 '/media/database/data4/hj/dataset/SMIC/smicM3',
                 '/media/database/data4/hj/dataset/SMIC/smicM4',
                 '/media/database/data4/hj/dataset/SMIC/smicM5',
                 ]
    train_dataset = casme2DataSet(casme2_path, subid, phase='train', transform=TwoCropTransform(data_transforms))
    print('subid:',subid)
    print('Train set size:', train_dataset.__len__())
    
    # print('data',train_dataset)  #返回的是getitem中的内容，的是图片的tensor，lable，idx
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    # 加载test data
    # data_transforms_val = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224,224)),# HW
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #     #                      std=[0.229, 0.224, 0.225])
    # ])
    #
    #
    #
    # test_dataset=casme2DataSet(casme2_path, subid, phase='test',
    #                                       transform=data_transforms_val)
    # test_loader=torch.utils.data.DataLoader(test_dataset,
    #                                                    batch_size=args.batch_size,
    #                                                    num_workers=args.workers,
    #                                                    shuffle=False,
    #                                                    pin_memory=True)
    #
    # print('Test set size:',  test_dataset.__len__())



    return train_loader
    # return train_dataset, test_dataset



