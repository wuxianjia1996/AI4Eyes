import os
import torch
import torch.nn as nn
import dataloader
import utils
import json
import torchvision.transforms as transforms
import argparse
import torchvision.models as models


parser = argparse.ArgumentParser(description='PyTorch Classification',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--test-dir', type = str, default = 'D:/OneDrive - whu.edu.cn\python/python_work/eye_images-main/cataract/test', help='test image directory')
parser.add_argument('--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--nb-cls', type = int, default=2, help='nb of output classes')
parser.add_argument('--inet-pretrain', action='store_true', help='imageNet pretrained')

args = parser.parse_args()
args.out_dir = 'D:\OneDrive - whu.edu.cn\python\python_work\eye_images-main\cataract'

# args.cuda = torch.cuda.is_available()

new_net = models.resnet18(pretrained= args.inet_pretrain)
new_net.fc = nn.Linear(512, args.nb_cls)
# checkpoint = torch.load('net_cataract.pt', map_location=lambda storage, loc: storage.cuda(0))
new_net.load_state_dict(torch.load('net_cataract.pt'))
# new_net.eval()

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

for (inputs, targets) in testloader:
    inputs = inputs.cuda(0) if args.cuda else inputs
    targets = targets.cuda(0) if args.cuda else targets
    outputs = new_net(inputs)
    result = utils.f1_score(outputs, targets, topk=(1,))
    errors = utils.get_back_errors(outputs, targets, topk=(1,))

    print(result)
    msg = 'Test accuracy: {:.3f} | precision: {:.3f} | recall: {:.3f} | F1_score: {:.3f}'.format(result[0],result[1],result[2],result[3])
    logger.info(msg)

    print(errors)
    msg = 'Test errors: {}'.format(errors)
    logger.info(msg)