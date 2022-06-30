import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from PIL import Image
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser(description='PyTorch Classification',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--nb-cls', type = int, default=2, help='nb of output classes')
parser.add_argument('--inet-pretrain', action='store_true', help='imageNet pretrained')

args = parser.parse_args()

# args.cuda = torch.cuda.is_available()

new_net = models.resnet18(pretrained= args.inet_pretrain)
new_net.fc = nn.Linear(512, args.nb_cls)
# checkpoint = torch.load('net_retinal.pt', map_location=lambda storage, loc: storage.cuda(0))
new_net.load_state_dict(torch.load('net_retinal.pt'))


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def draw_cam(model, img_path, save_path, transform=None, visheadmap=False):
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
    features = x
    # print(features.shape)
    output = model.avgpool(x)
    # print(output.shape)
    output = output.view(output.size(0), -1)
    # print(output.shape)
    output = model.fc(output)
    # print(output.shape)

    def extract(g):
        global feature_grad
        feature_grad = g

    pred = torch.argmax(output).item()
    print(pred)
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)

    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()

    img = cv2.imread(img_path)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255 * headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap * 0.4 + img
    cv2.imwrite(save_path, superimposed_img)


if __name__ == '__main__':
    model = new_net
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    img_path = './detach_retinal/0.jpg'
    save_path = './detach_retinal/cam_0.png'
    draw_cam(model, img_path, save_path, transform=transform, visheadmap=True)