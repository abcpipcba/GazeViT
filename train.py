import argparse
import math
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import datetime

import numpy as np
from dataset.CVACT import CVACT
from model.GazeViT import GazeViT
from criterion.soft_triplet import SoftTripletBiLoss
from criterion.sam import SAM

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.03, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--op', default='adam', type=str,
                    help='sgd, adam, adamw')

parser.add_argument('--attention', default=True, action='store_true',
                    help='hybrid attention')

parser.add_argument('--sat_res', default='', type=int,
                    help='enlarge resolution')

parser.add_argument('--crop', default=True, action='store_true',
                    help='image crop')

best_acc1 = 0

args = parser.parse_args()
print(args)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mean_ps = AverageMeter('Mean-P', ':6.2f')
    mean_ns = AverageMeter('Mean-N', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mean_ps, mean_ns],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images_s, images_a, indexes, _, delta, atten, _, heatmap) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        images_s = images_s.cuda()
        images_a = images_a.cuda()
        indexes = indexes.cuda()

        # compute output
        if args.crop:
            embed_s, embed_a = model(im_s=images_s, im_a=images_a, delta=delta, atten=atten, heatmap=heatmap)
        else:
            embed_s, embed_a = model(im_s =images_s, im_a=images_a, delta=delta)

        loss, mean_p, mean_n = criterion(embed_s, embed_a)

        losses.update(loss.item(), images_s.size(0))
        mean_ps.update(mean_p, images_s.size(0))
        mean_ns.update(mean_n, images_s.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        if args.op != 'sam':
            optimizer.step()
        else:
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            if args.crop:
                embed_s, embed_a = model(im_q=images_s, im_k=images_a, delta=delta, atten=atten)
            else:
                embed_s, embed_a = model(im_q=images_s, im_k=images_a, delta=delta)

            loss, mean_p, mean_n = criterion(embed_s, embed_a)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        del loss
        del embed_s
        del embed_a

def validate(val_street_loader, val_aerial_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress_q = ProgressMeter(
        len(val_street_loader),
        [batch_time],
        prefix='Test_street: ')
    progress_k = ProgressMeter(
        len(val_aerial_loader),
        [batch_time],
        prefix='Test_aerial: ')

    model_street = model.module.street_net
    model_aerial = model.module.aerial_net
    
    model_street.cuda()
    model_aerial.cuda()

    model_street.eval()
    model_aerial.eval()

    street_features = np.zeros([len(val_street_loader.dataset), args.dim])
    street_labels = np.zeros([len(val_street_loader.dataset)])
    aerial_features = np.zeros([len(val_aerial_loader.dataset), args.dim])

    with torch.no_grad():
        end = time.time()
        
        for i, (images, indexes, atten) in enumerate(val_aerial_loader):
            
            images = images.cuda()
            indexes = indexes.cuda()

            # compute output
            if args.crop:
                aerial_embed = model_aerial(x=images, atten=atten)
            else:
                aerial_embed = model_aerial(x=images, indexes=indexes)  # delta

            aerial_features[indexes.cpu().numpy().astype(int), :] = aerial_embed.detach().cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_k.display(i)

        end = time.time()

        for i, (images, indexes, labels) in enumerate(val_street_loader):
            
            images = images.cuda()
            indexes = indexes.cuda()
            labels = labels.cuda()

            # compute output
            street_embed = model_street(images)

            street_features[indexes.cpu().numpy(), :] = street_embed.detach().cpu().numpy()
            street_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_q.display(i)

        [top1, top5, result_list] = accuracy(street_features, aerial_features, street_labels.astype(int))

    return top1, result_list

def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # taken from https: //github.com/Jeff-Zilence/TransGeo2022
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVACT and CVUSA
    query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
    reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
    similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

    for i in range(N):
        ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    
    return results[0], results[1], results

# save all the attention map
def scan(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Scan:")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_s, images_k, _, indexes , delta, _, _, _) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            images_s = images_s.cuda()
            images_k = images_k.cuda()
            indexes = indexes.cuda()

            # compute output
            embed_s, embed_k = model(im_q =images_s, im_k=images_k, delta=delta, indexes=indexes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

model = GazeViT(args=args)
model = torch.nn.DataParallel(model).cuda()
criterion = SoftTripletBiLoss().cuda()

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
base_optimizer = torch.optim.AdamW
optimizer = SAM(parameters, base_optimizer,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
    os.mkdir(os.path.join(args.save_path, 'attention'))
    os.mkdir(os.path.join(args.save_path, 'attention','train'))
    os.mkdir(os.path.join(args.save_path, 'attention','val'))

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# freeze      
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad_(False)
        
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        
        checkpoint = torch.load(args.resume)
        if not args.crop:
            args.start_epoch = checkpoint['epoch']
        if args.crop and args.sat_res != 0:
            pos_embed_reshape = checkpoint['state_dict']['module.aerial_net.pos_embed'][:, 2:, :].reshape(
                [1,
                 np.sqrt(checkpoint['state_dict']['module.aerial_net.pos_embed'].shape[1] - 2).astype(int),
                 np.sqrt(checkpoint['state_dict']['module.aerial_net.pos_embed'].shape[1] - 2).astype(int),
                 model.module.aerial_net.embed_dim]).permute((0, 3, 1, 2))
            checkpoint['state_dict']['module.aerial_net.pos_embed'] = \
                torch.cat([checkpoint['state_dict']['module.aerial_net.pos_embed'][:, :2, :],
                           torch.nn.functional.interpolate(pos_embed_reshape, (
                           args.sat_res // model.module.aerial_net.patch_embed.patch_size[0],
                           args.sat_res // model.module.aerial_net.patch_embed.patch_size[1]),
                                                           mode='bilinear').permute((0, 2, 3, 1)).reshape(
                               [1, -1, model.module.aerial_net.embed_dim])], dim=1)

        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

dataset = CVACT
train_dataset = dataset(mode='train', args=args)
val_scan_dataset = dataset(mode='scan_val', args=args)
val_street_dataset = dataset(mode='test_street', args=args)
val_aerial_dataset = dataset(mode='test_aerial', args=args)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

train_scan_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True,sampler=None, drop_last=False)

val_scan_loader = torch.utils.data.DataLoader(
    val_scan_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True,
    sampler=None, drop_last=False)

val_street_loader = torch.utils.data.DataLoader(
    val_street_dataset,batch_size=32, shuffle=False,
    num_workers=args.workers, pin_memory=True) # 512, 64
val_aerial_loader = torch.utils.data.DataLoader(
    val_aerial_dataset, batch_size=64, shuffle=False,
    num_workers=args.workers, pin_memory=True) # 80, 128

# validate(val_street_loader, val_aerial_loader, model, args)

result_list = []
for epoch in range(args.start_epoch, args.epochs):

    print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
    
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args, train_sampler = None)

    acc1, results = validate(val_street_loader, val_aerial_loader, model, args)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    result_list.append(results[:4])
    
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(args.save_path,'checkpoint.pth.tar'))
    
    if args.crop:
        model.module.aerial_net.save = os.path.join(args.save_path, 'attention', 'train')
        scan(train_scan_loader, model, args)
        model.module.aerial_net.save = os.path.join(args.save_path, 'attention', 'val')
        scan(val_scan_loader, model, args)
        
        model.module.aerial_net.save = None
