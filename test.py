import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import utils
import json
import torchvision.transforms as transforms
import argparse
import torchvision.models as models
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='PyTorch Classification',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--test-dir', type = str, default = 'test_dataset path', help='test image directory')
parser.add_argument('--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--nb-cls', type = int, default=2, help='nb of output classes')
parser.add_argument('--inet-pretrain', action='store_true', help='imageNet pretrained')

args = parser.parse_args()
args.out_dir = 'output path'
os.makedirs(args.out_dir, exist_ok=True)

args.cuda = torch.cuda.is_available()

new_net = models.resnet18(pretrained= args.inet_pretrain)
new_net.fc = nn.Linear(512, args.nb_cls)
# checkpoint = torch.load('net_cataract.pt', map_location=lambda storage, loc: storage.cuda(0))
new_net.load_state_dict(torch.load('checkpoint_path'))
new_net.eval()
new_net.to('cuda:0')

nb_img = 0
for root, dirs, files in os.walk(args.test_dir):
    for each in files:
        nb_img += 1

logger = utils.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
test_transform = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,])

testloader = dataloader.TestLoader(nb_img, args.test_dir, test_transform)

losses = utils.AverageMeter()
top1 = utils.AverageMeter()

preds = []
auc_preds = []
true_label = []

for (inputs, targets) in testloader:
    inputs = inputs.cuda(0) if args.cuda else inputs
    targets = targets.cuda(0) if args.cuda else targets
    outputs = new_net(inputs)
    auc_pred = F.softmax(outputs, dim=1)   # 计算auc
    pred = torch.argmax(outputs, dim=1)    # 计算分类metrics

    auc_preds.append(auc_pred)
    preds.append(pred)
    true_label.append(targets)

    # result = utils.f1_score(outputs, targets, topk=(1,))
    # errors = utils.get_back_errors(outputs, targets, topk=(1,))
    #
    # print(result)
    # msg = 'Test accuracy: {:.3f} | precision: {:.3f} | recall: {:.3f} | F1_score: {:.3f}'.format(result[0],result[1],result[2],result[3])
    # logger.info(msg)
    #
    # print(errors)
    # msg = 'Test errors: {}'.format(errors)
    # logger.info(msg)

    acc1 = utils.accuracy(outputs, targets, topk=(1, ))
    top1.update(acc1[0].item(), inputs.size()[0])

    msg = 'acc1: {:.3f}%'.format(acc1[0].item())
    precision = precision_score(pred.detach().cpu().numpy(), targets.detach().cpu().numpy(), average="weighted")
    recall = recall_score(pred.detach().cpu().numpy(), targets.detach().cpu().numpy(), average="weighted")
    f1 = f1_score(pred.detach().cpu().numpy(), targets.detach().cpu().numpy(), average="weighted")

msg1 = 'Top1: {:.3f}%'.format(top1.avg)
print(args.test_dir, ": ", msg1)
logger.info(msg1)

combined_preds = torch.cat(preds, dim=0).detach().cpu().numpy()
auc_combined_preds = torch.cat(auc_preds, dim=0).detach().cpu().numpy()
combined_true_label = torch.cat(true_label, dim=0).detach().cpu().numpy()

precision_ave = precision_score(combined_preds, combined_true_label, average="weighted")
recall_ave = recall_score(combined_preds, combined_true_label, average="weighted")
f1_ave = f1_score(combined_preds, combined_true_label, average="weighted")

msg2 = '"F1 score average:", {:.3f}%'.format(f1_ave)
msg3 = '"precision average:", {:.3f}%'.format(precision_ave)
msg4 = '"recall average:", {:.3f}%'.format(recall_ave)
print(args.test_dir, ' ', msg2)
print(args.test_dir, ' ', msg3)
print(args.test_dir, ' ', msg4)
logger.info(msg2)
logger.info(msg3)
logger.info(msg4)

# 计算二分类auc
auc = roc_auc_score(combined_true_label, auc_combined_preds[:, 1])
print("auc: ", auc)

# 计算多分类auc
lb = LabelBinarizer()
lb.fit(combined_true_label)
y_true_one_hot = lb.transform(combined_true_label)

auc_scores = []
for i in range(args.nb_cls):
    auc = roc_auc_score(y_true_one_hot[:, i], auc_combined_preds[:, i])
    auc_scores.append(auc)

macro_auc = np.mean(auc_scores)
auc_score = roc_auc_score(combined_preds, combined_true_label, average='macro')
print("AUC Score: ", macro_auc)


# 计算每个类别的ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve((combined_true_label == i), combined_preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()

# 四分类
colors = ['aqua', 'darkorange', 'cornflowerblue', 'pink']

for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, lable='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')

plt.savefig('four_class_roc.png')
plt.show()
