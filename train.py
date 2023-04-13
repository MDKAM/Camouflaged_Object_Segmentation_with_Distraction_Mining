import datetime
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import training_root
from config import backbone_path
from datasets import ImageFolder
from avgMeter import measure_avg
from PFNet import PFNet
cudnn.benchmark = True
torch.manual_seed(2021)
device_ids = [1]
output_path = './output'
net_name = 'PFNet'

args = {
    'epoch_num': 45,
    'train_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'SGD',
}

# print(torch.__version__)

# Path making
def check_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

check_directory(output_path)
check_directory(os.path.join(output_path, net_name))
vis_path = os.path.join(output_path, net_name, 'log')
check_directory(vis_path)
log_path = os.path.join(output_path, net_name, str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + '.txt')
# print(datetime.datetime.now())
writer = SummaryWriter(log_dir=vis_path, comment=net_name)

# Transform Data in order to have all variations of data in training process
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare training Data Set
train_set = ImageFolder(training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
# ########################## iou loss #############################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - intersection
        iou = 1 - (intersection / union)
        return iou.mean()


# #################### structure loss #############################
class Structure_Loss(torch.nn.Module):
    def __init__(self):
        super(Structure_Loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)


structure_loss = Structure_Loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = IOU().cuda(device_ids[0])

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def main():
    print(args)
    print(net_name)

    net = PFNet(backbone_path).cuda(device_ids[0]).train()

    print("SGD")
    #different laerning rates for bias and weight parameters
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    t_cur = 1
    # start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record = measure_avg()
        loss_1_record = measure_avg()
        loss_2_record = measure_avg()
        loss_3_record = measure_avg()
        loss_4_record = measure_avg()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(t_cur) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4 = net(inputs)

            loss_1 = bce_iou_loss(predict_1, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_4 = structure_loss(predict_4, labels)

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)

            if t_cur % 10 == 0:
                writer.add_scalar('loss', loss, t_cur)
                writer.add_scalar('loss_1', loss_1, t_cur)
                writer.add_scalar('loss_2', loss_2, t_cur)
                writer.add_scalar('loss_3', loss_3, t_cur)
                writer.add_scalar('loss_4', loss_4, t_cur)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, t_cur, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                   loss_3_record.avg, loss_4_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            t_cur += 1

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(output_path, net_name, '%d.pth' % epoch))
            print("Optimization Have Done!")
            return

if __name__ == '__main__':
    main()