import os
import torch
import torch.nn as nn
import json
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F
import argparse
import torchvision.models as models
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
new_net.load_state_dict(torch.load('net_cataract.pt'))
final_conv = 'layer4'
new_net.eval()
new_net.cuda()

classes = {0: 'normal', 1: 'cataract'}

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().detach().numpy())

new_net._modules.get(final_conv).register_forward_hook(hook_feature)

# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    print(bz, nc, h, w)
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    print(h_x)
    probs, idx = h_x.sort(0, True)
    print(probs, idx)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    # cv2.imwrite('./cataract/heatmap.jpg', heatmap)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('./cataract/cam1368.jpg', result)


if __name__ == '__main__':
    model = new_net
    root = './cataract/1368.jpg'
    img = Image.open(root)
    get_cam(model, features_blobs, img, classes, root)

