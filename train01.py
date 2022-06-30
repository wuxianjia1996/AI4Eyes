import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

import json
import os
import argparse

import utils
import dataloader




# Warm up: lr from 1e-8 to max-lr
def warmup(trainloader, net, criterion, optimizer, logger, args): 
    
    net.train()
    nb_iter = 0
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    while nb_iter < args.warmup_iter:

       for (inputs, targets) in trainloader:
        
            inputs = inputs.cuda() if args.cuda else inputs
            targets = targets.cuda() if args.cuda else targets
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            nb_iter += 1
            lr = utils.warmup_learning_rate(optimizer, nb_iter, args.warmup_iter, args.max_lr)
            acc1 = utils.accuracy(outputs, targets, topk=(1, ))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(acc1[0].item(), inputs.size()[0])
            if nb_iter % 100 == 0:
                msg = 'Warmup iter: {:d} | Lr {:.7f} | Loss: {:.3f} | Top1: {:.3f}% '.format(nb_iter, lr, losses.avg, top1.avg)
                logger.info(msg)
                losses = utils.AverageMeter()
                top1 = utils.AverageMeter()
            if nb_iter == args.warmup_iter:
                break
            

def evaluate(valloader, net, logger, best_acc, args):
    net.eval()
    correct = 0
    total = 0
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()    
    for (inputs, targets) in valloader:
        inputs = inputs.cuda() if args.cuda else inputs
        targets = targets.cuda() if args.cuda else targets
        outputs = net(inputs)
        acc1 = utils.accuracy(outputs, targets, topk=(1, ))
        top1.update(acc1[0].item(), inputs.size()[0])
        
    msg = 'Evaluation -->>> Top1: {:.3f}% '.format(top1.avg)
    logger.info(msg)
    
    # Save checkpoint.
    acc = top1.avg
    if acc > best_acc:
        
        logger.info ('Saving Best')
        torch.save(net.state_dict(), os.path.join(args.out_dir, 'netBest.pth'))
        best_acc = acc

    logger.info ('Saving Last')
    torch.save(net.state_dict(), os.path.join(args.out_dir, 'netLast.pth'))

    msg = 'Best Performance: {:.3f}'.format(best_acc)
    logger.info(msg)
    return best_acc


                
        
parser = argparse.ArgumentParser(description='PyTorch Classification',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--max-lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--out-dir', type=str, help='output directory')

parser.add_argument('--train-dir', type = str, default = 'D:/OneDrive - whu.edu.cn\python/python_work/eye_images-main/cataract/train', help='train image directory')
parser.add_argument('--val-dir', type = str, default = 'D:/OneDrive - whu.edu.cn\python/python_work/eye_images-main/cataract/val', help='val image directory')
parser.add_argument('--batch-size', type = int, default = 64, help='batch size')
parser.add_argument('--total-iter', type = int, default = 500, help='nb of iterations')
parser.add_argument('--nb-cls', type = int, default=2, help='nb of output classes')
parser.add_argument('--warmup-iter', type = int, default=500, help='warm up iteration')
parser.add_argument('--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--resume-pth', type = str, help='resume_pth')
parser.add_argument('--inet-pretrain', action='store_true', help='imageNet pretrained')


args = parser.parse_args()
args.out_dir = 'D:\OneDrive - whu.edu.cn\python\python_work\eye_images-main\cataract'

args.cuda = torch.cuda.is_available()

#if not os.path.isdir(args.out_dir):
#    os.mkdir(args.out_dir)
    
logger = utils.get_logger(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

best_acc = 0  # best test accuracy

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,])
val_transform = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,])

trainloader = dataloader.TrainLoader(args.batch_size, args.train_dir, train_transform)
valloader = dataloader.ValLoader(args.batch_size, args.val_dir, val_transform)

randomSeed = 123
torch.backends.cudnn.deterministic = True
torch.manual_seed(randomSeed)

net = models.resnet18(pretrained= args.inet_pretrain)
net.fc = nn.Linear(512, args.nb_cls)

if args.resume_pth:
    net.load_state_dict(torch.load(args.resume_pth))
    msg = 'Loading weight from {}'.format(args.resume_pth)
    logger.info (msg)


if args.cuda:
    net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), 1e-8, weight_decay=args.weight_decay)

## remove grad for evaluation 
with torch.no_grad() : 
    best_acc = evaluate(valloader, net, logger, best_acc, args)


warmup(trainloader, net, criterion, optimizer, logger, args)

## remove grad for evaluation 
with torch.no_grad() : 
    best_acc = evaluate(valloader, net, logger, best_acc, args)
    
net.train()
nb_iter = 0
losses = utils.AverageMeter()
top1 = utils.AverageMeter()

while nb_iter < args.total_iter : 

    for (inputs, targets) in trainloader:

        inputs = inputs.cuda() if args.cuda else inputs
        targets = targets.cuda() if args.cuda else targets
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        nb_iter += 1
        lr = utils.cos_learning_rate(optimizer, nb_iter, args.total_iter, args.max_lr, min_lr = 1e-8)
        
        acc1 = utils.accuracy(outputs, targets, topk=(1, ))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(acc1[0].item(), inputs.size()[0])
        if nb_iter % 100 == 0:
            msg = 'training iter: {:d} | Lr {:.7f} | Loss: {:.3f} | Top1: {:.3f}%'.format(nb_iter, lr, losses.avg, top1.avg)
            logger.info(msg)
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            with torch.no_grad() :
                best_acc = evaluate(valloader, net, logger, best_acc, args)
            net.train()
            
        if nb_iter == args.total_iter : 
            break


torch.save(net.state_dict(), 'net_cataract.pt')


msg = 'mv {0} {1}'.format(os.path.join(args.out_dir, 'netBest.pth'), os.path.join(args.out_dir, 'netBest{:.3f}.pth'.format(best_acc)))
logger.info(msg)
# os.system(msg)
