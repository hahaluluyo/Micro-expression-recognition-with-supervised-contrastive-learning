import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from resnet import SupConResNet, LinearClassifier

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def draw_cam(model,classifier, img_path, save_path, transform=None, visheadmap=False):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    x = model.conv1(img)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = x  # 1x2048x7x7
    avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    fc=torch.nn.Linear(512,5)
    output = avgpool(x)
    output = output.view(output.size(0), -1)
    output=fc(output)

    # print(features.shape)
    # output = model.avgpool(x)  # 1x2048x1x1
    # print(output.shape)
    # output = output.view(output.size(0), -1)
    # print(output.shape)  # 1x2048
    # output = model.fc(output)  # 1x1000
    # print(output.shape)
    #output=classifier(model(img).detach())

    def extract(g):
        global feature_grad
        feature_grad = g

    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)# 注册一个钩子,当代码运行到涉及梯度（比如backward）时，会想起这里的钩子，并执行传入的函数hook_fn，其参数grad就是feature梯度，可以将这个梯度保存下来
    pred_class.backward()
    greds = feature_grad# # 计算梯度，计算到features特征层时，触发钩子，执行extract，并将梯度赋值给全局变量features_grad，方便在函数外面获取数值
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap = np.maximum(headmap, 0)
    headmap /= np.max(headmap)

    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()
    # 融合类激活图和原始图片
    img = cv2.imread(img_path)
    #img=cv2.resize(img,(224,224))
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255 * headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap * 0.4 + img *0.5
    cv2.imwrite(save_path, superimposed_img)


if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    # model=SupConResNet('resnet18')
    classifier = LinearClassifier('resnet18', 5)
    # ckpt = torch.load(
    #     './save/SupCon/casme2_models/SupCon_casme2_resnet18_lr_0.0001_decay_0.0001_bsz_20_temp_0.07_trial_0/last.pth',
    #     map_location='cpu')
    # state_dict = ckpt['model']
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     k = k.replace("module.", "")
    #     new_state_dict[k] = v
    #     state_dict = new_state_dict
    # model.load_state_dict(state_dict)
    # model=model.encoder

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    draw_cam(model, classifier,'/media/database/data4/hj/dataset/casM3/000148-vis.png', './heatmap_result/base/148.png', transform=transform, visheadmap=True)
