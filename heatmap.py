import interpretdl as it
from resnet import ResNet,SupConResNet
from PIL import Image
from interpretdl.data_processor.readers import read_image
from torch.utils.tensorboard import SummaryWriter
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch import nn
import itertools
from sklearn.metrics import confusion_matrix
savepath ='./save'
if not os.path.exists(savepath):
    os.mkdir(savepath)


def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))

class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        #model_ft = ResNet('resnet18', 5, pretrained='./resnet18.pth').resnet_base
        model = SupConResNet('resnet18')
        ckpt = torch.load('./save/SupCon/casme2_models/SupCon_casme2_resnet18_lr_0.1_decay_0.0001_bsz_20_temp_0.07_trial_0/last.pth',
                          map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)

        self.model = model.resnet_base

    def forward(self, x):
        if True:  # draw features or not
            x = self.model.conv1(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f1_conv1.png".format(savepath))

            x = self.model.bn1(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f2_bn1.png".format(savepath))

            x = self.model.relu(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))

            x = self.model.maxpool(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.model.layer1(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

            x = self.model.layer2(x)
            draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

            x = self.model.layer3(x)
            draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

            x = self.model.layer4(x)
            draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))

            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            plt.savefig("{}/f10_fc.png".format(savepath))
            plt.clf()
            plt.close()


        return x

def heat_map():
        model = ft_net().cuda()

        # pretrained_dict = resnet50.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # net.load_state_dict(model_dict)
        model.eval()
        img = cv2.imread('./000008-vis.png')
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = transform(img).cuda()
        img = img.unsqueeze(0)

        with torch.no_grad():
            start = time.time()
            out = model(img)
            print("total time:{}".format(time.time() - start))
            result = out.cpu().numpy()
            # ind=np.argmax(out.cpu().numpy())
            ind = np.argsort(result, axis=1)
            for i in range(5):
                print("predict:top {} = cls {} : score {}".format(i + 1, ind[0, 1000 - i - 1], result[0, 1000 - i - 1]))
            print("done")
# 绘制混淆矩阵
def plt_confusion_matrix(cm, classes=  ['Happiness', 'Repression', 'Surprise','Disgust', 'Others'],
                         normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):#OrRd#Blues
    """
    casme2_5:
    classes= ['Happiness','Repression', 'Surprise', 'Disgust', 'Others'],
    casme2_3:
    classes= ['Negative', 'Positive', 'Surprise']
    samm_5:
    ['Anger','Contempt', 'Happiness', 'Surprise','Others']
    samm_3:

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15,rotation=45)
    plt.yticks(tick_marks, classes,fontsize=15)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #设置字体颜色和大小
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=13)#cas5=10
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.21)
    plt.figure(dpi=600)# 分辨率

    plt.show()
def gradcam():
    model = SupConResNet('resnet18')
    ckpt = torch.load(
        './save/SupCon/casme2_models/SupCon_casme2_resnet18_lr_0.1_decay_0.0001_bsz_20_temp_0.07_trial_0/last.pth',
        map_location='cpu')
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)

    model = model.resnet_base
    target_layers = [model.layer4[-1]]

    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    img = cv2.imread('./000008-vis.png')
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img).cuda()
    input_tensor = img.unsqueeze(0)


    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(281)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

if __name__=='__main__':
        #heat_map()
        matrix = np.array(     [[19,  1 , 1  ,2 , 9],
 [ 5  ,12 , 0,  0 ,10],
 [ 0 , 0 ,25,  0 , 0],
 [ 2 , 0 , 0, 55 , 6],
 [ 5 , 0,  2, 22 ,70]]

)
        M=matrix
        # n = len(M)
        # for i in range(n):
        #     rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
        #     try:
        #         print(
        #         'precision: %s' % (M[i][i] / float(colsum)), 'recall: %s' % (M[i][i] / float(rowsum)))
        #     except ZeroDivisionError:
        #         print('precision: %s' % 0, 'recall: %s' % 0)

        plt_confusion_matrix(matrix,  normalize=True)
