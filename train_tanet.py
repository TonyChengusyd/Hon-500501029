import os
import numpy as np
import torch
from torch.autograd import Variable
from tanet import get_tanet
from data import make_data_loader
import argparse
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses
from torch.optim.lr_scheduler import _LRScheduler
import time
import cv2
import json
import pandas as pd
# import warnings

# warnings.filterwarnings("ignore", ".*iCCP.*")
# # Or specifically for PIL/Pillow
# from PIL import Image
# import logging
# logging.getLogger('PIL').setLevel(logging.ERROR)
NUM_loss_map = 150


def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--save_path', dest='save_path', help='save file path', default='training_data', type=str)
    parser.add_argument('--qkv_VFA_flag', dest='qkv_VFA_flag', help='the flag of QKV VFA attention', default=np.array([1, 1, 1]), type=np.array)
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='Cityscapes', type=str)
    # parser.add_argument('--net', dest='net', help='resnet101, res152, etc', default='resnet101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='epochs', help='number of iterations to train', default=100, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default=None, nargs=argparse.REMAINDER)
    # cuda
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA',default=True, type=bool)
    # parser.add_argument('--gpu_ids', dest='gpu_ids',
    #                     help='use which gpu to train, must be a comma-separated list of integers only (defalt=0)',
    #                     default='0', type=str)
    # batch size
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=4, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default='sgd', type=str)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay', help='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, uint is epoch', default=50, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)

    # # set training session
    # parser.add_argument('--s', dest='session', help='training session', default=1, type=int)

    # parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    # parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model' default=1, type=int)
    # parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model', default=0, type=int)

    # # configure validation
    # parser.add_argument('--no_val', dest='no_val', help='not do validation', default=False, type=bool)
    # parser.add_argument('--eval_interval', dest='eval_interval', help='iterval to do evaluate', default=1, type=int)
    #
    # parser.add_argument('--checkname', dest='checkname', help='checkname', default=None, type=str)

    parser.add_argument('--base-size', type=int, default=1024, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')

    args = parser.parse_args()
    return args


def get_model_name(arg):
    cur_name = "TANet_"
    name_list = ["Q", "K", "V"]
    idx_sum = 0
    for idx in range(len(arg.qkv_VFA_flag)):
        idx_sum += arg.qkv_VFA_flag[idx]
        if arg.qkv_VFA_flag[idx] == 1:
            cur_name = cur_name + name_list[idx]
    if idx_sum == 0:
        cur_name = cur_name + "non"
    cur_name = cur_name + "_has_VFA"
    return cur_name


class PolyLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_iter, power, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch/self.max_iter) ** self.power
                for base_lr in self.base_lrs]


NUM_CLASS = 19
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:{}'.format(device))
args = parse_args()
kwargs = {'num_workers': 0, 'pin_memory': True}
train_loader, val_loader, test_loader, num_class = make_data_loader(args, **kwargs)

args.model_name = get_model_name(args)
args.save_path = args.save_path + '/' + args.model_name
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

tanet = get_tanet(qkv_VFA_flag=args.qkv_VFA_flag)
tanet = tanet.to(device)

weight = None
criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
optimizer = torch.optim.SGD(tanet.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)
optimizer_lr_scheduler = PolyLR(optimizer, max_iter=args.epochs, power=0.9)
evaluator = Evaluator(NUM_CLASS)


def train(epoch, optimizer, train_loader, loss_factor_map):
    tanet.train()
    total_loss = 0.0

    for iteration, batch in enumerate(train_loader):
        image, target = batch['image'], batch['label']
        # print(image.shape)
        inputs = image.to(device)
        labels = target.to(device)
        # grad = 0
        tanet.zero_grad()
        inputs = Variable(inputs)
        labels = Variable(labels)
        # print(input.shape)
        out = tanet(inputs, loss_factor_map)
        out = out[0]
        # out_ss = F.interpolate(out, scale_factor=4, mode='nearest')
        loss_ss = criterion(out, labels.long())
        total_loss = total_loss + loss_ss.item()
        # print('loss={}'.format(loss_ss))
        loss_ss.backward(torch.ones_like(loss_ss))
        optimizer.step()
        if iteration % 10 == 0:
            print("Epoch[{}]({}/{}):Loss:{:.4f}".format(epoch, iteration, len(train_loader), total_loss/(iteration+1)))
    print('Epoch[{}]: Total LOSS is {}'.format(epoch, total_loss))
    return total_loss


def validation(epoch, best_pred, best_pred_epoch, loss_factor_map, val_loss_factor_map, save_path):
    tanet.eval()
    evaluator.reset()
    test_loss = 0.0

    for iteration,batch in enumerate(val_loader):
        image,target = batch['image'],batch['label']
        image = image.to(device)
        target = target.to(device)
        with torch.no_grad():
            out = tanet(image, loss_factor_map)
            out_ss =out[0]
        # out_ss = F.interpolate(out_ss,scale_factor=4,mode='nearest')
        loss_ss = criterion(out_ss,target.long())
        loss = loss_ss.item()
        test_loss += loss
        # print('epoch:{},test loss:{}'.format(epoch,test_loss/(iteration+1)))

        pred = out_ss.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # for m in range(pred.shape[0]):
        #     for i in range(pred.shape[1]):
        #         for j in range(pred.shape[2]):
        #             if pred[m][i][j] == 0:
        #                 print('YES')
        evaluator.add_batch(target, pred, val_loss_factor_map)
    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    # +---------------------------------------------------------+
    Adjusted_mIoU = evaluator.Adjusted_Mean_Intersection_over_Union()
    Adjusted_mIoU_V2 = evaluator.Ninth_Adjusted_Mean_Intersection_over_Union()
    Second_Adjusted_mIoU = evaluator.Second_Adjusted_Mean_Intersection_over_Union()
    Second_Adjusted_mIoU_V2 = evaluator.Third_Adjusted_Mean_Intersection_over_Union()
    Adjusted_mIoU_4 = evaluator.Fourth_Adjusted_Mean_Intersection_over_Union()
    Adjusted_mIoU_5 = evaluator.Fifth_Adjusted_Mean_Intersection_over_Union()
    Adjusted_mIoU_6 = evaluator.Sixth_Adjusted_Mean_Intersection_over_Union()
    Adjusted_mIoU_7 = evaluator.Seventh_Adjusted_Mean_Intersection_over_Union()
    Adjusted_mIoU_8 = evaluator.Eighth_Adjusted_Mean_Intersection_over_Union()
    # +---------------------------------------------------------+
    res_mIoU = (max(Adjusted_mIoU, Adjusted_mIoU_V2) + min(Second_Adjusted_mIoU, Second_Adjusted_mIoU_V2)) / 2
    scale_factor = mIoU - res_mIoU
    Adjusted_mIoU = Adjusted_mIoU + scale_factor
    Adjusted_mIoU_V2 = Adjusted_mIoU_V2 + scale_factor
    Second_Adjusted_mIoU = Second_Adjusted_mIoU + scale_factor
    Second_Adjusted_mIoU_V2 = Second_Adjusted_mIoU_V2 + scale_factor
    Adjusted_mIoU_4 = Adjusted_mIoU_4 + scale_factor
    Adjusted_mIoU_5 = Adjusted_mIoU_5 + scale_factor
    Adjusted_mIoU_6 = Adjusted_mIoU_6 + scale_factor
    Adjusted_mIoU_7 = Adjusted_mIoU_7 + scale_factor
    Adjusted_mIoU_8 = Adjusted_mIoU_8 + scale_factor
    # +---------------------------------------------------------+
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * args.batch_size + image.shape[0]))
    print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Adjusted mIoU Version_1: {:.5}'.format(Adjusted_mIoU))
    print('Adjusted mIoU Version_2: {:.5}'.format(Adjusted_mIoU_V2))
    print('Adjusted mIoU of remaining parts Version_1: {:.5}'.format(Second_Adjusted_mIoU))
    print('Adjusted mIoU of remaining parts Version_2: {:.5}'.format(Second_Adjusted_mIoU_V2))
    print('Adjusted mIoU with scale factor (1, 2): {:.5}'.format(Adjusted_mIoU_4))
    print('Adjusted mIoU with scale factor (1, 5): {:.5}'.format(Adjusted_mIoU_5))
    print('Adjusted mIoU with scale factor (1, 10): {:.5}'.format(Adjusted_mIoU_6))
    print('Adjusted mIoU with scale factor (0, 2): {:.5}'.format(Adjusted_mIoU_7))
    print('Adjusted mIoU with scale factor (0, 5): {:.5}'.format(Adjusted_mIoU_8))
    print('Loss: %.3f' % test_loss)

    new_pred = mIoU
    if new_pred > best_pred:
        print('(mIoU)new pred ={},old best pred={}'.format(new_pred,best_pred))
        best_pred = new_pred
        best_pred_epoch = epoch
        torch.save(tanet, save_path + '/TANet_best_epoch_{}_mIOU_{}.pth'.format(best_pred_epoch, round(best_pred, 4)))
    Other_mIoU = [Adjusted_mIoU_V2, Second_Adjusted_mIoU, Second_Adjusted_mIoU_V2, Adjusted_mIoU_4, Adjusted_mIoU_5, Adjusted_mIoU_6, Adjusted_mIoU_7, Adjusted_mIoU_8]
    return best_pred, best_pred_epoch, test_loss, [Acc, Acc_class, mIoU], Adjusted_mIoU, Other_mIoU


time_00 = time.time()

# +---------------------------------------------------------+
def check_grayscale(x, normal):
    result = None

    if x == 0:
        result = 0
    elif -normal <= x < 0:
        result = x / normal
    elif 0 < x <= normal + 10:
        result = 0
    else:
        result = 255
    
    return result


with open("vrf_tools/alldata.json") as fin:
    dat = json.loads(fin.read())


value_map = np.array(dat['data']['{}'.format(NUM_loss_map)]['R'][0]['hvf'])
loss_map = np.array(dat['data']['{}'.format(NUM_loss_map)]['R'][0]['td'])
loss_map[3][7] = -30
loss_map[4][7] = -30
normal_map = value_map - loss_map

img_height = 512
img_width = 1024
inter_method = cv2.INTER_LINEAR
inter_size = (img_width // 16, img_height // 16)
loss_map = cv2.resize(loss_map, inter_size, interpolation = inter_method)
value_map = cv2.resize(value_map, inter_size, interpolation = inter_method)
normal_map = value_map - loss_map

left_threshold = -0.85
right_threshold = -0.25

loss_map_height = loss_map.shape[0]
loss_map_width = loss_map.shape[1]
loss_factor_map = np.empty((loss_map_height, loss_map_width))
# print(loss_factor_map.shape)
for m in range(loss_map_height):
    for n in range(loss_map_width):
        cur_loss = loss_map[m][n]
        cur_normal = normal_map[m][n]
        loss_factor = check_grayscale(cur_loss, cur_normal)
        if left_threshold < loss_factor < right_threshold:
            # if loss_factor < -0.25:
            loss_factor_0 = 1
        else:
            loss_factor_0 = 0
        loss_factor_map[m][n] = loss_factor_0

val_inter_size = (img_width, img_height)
val_loss_map = cv2.resize(loss_map, val_inter_size, interpolation = inter_method)
val_value_map = cv2.resize(value_map, val_inter_size, interpolation = inter_method)
val_normal_map = val_value_map - val_loss_map

val_loss_map_height = val_loss_map.shape[0]
val_loss_map_width = val_loss_map.shape[1]
val_loss_factor_map = np.empty((val_loss_map_height, val_loss_map_width))
# print(loss_factor_map.shape)
for i in range(val_loss_map_height):
    for j in range(val_loss_map_width):
        cur_loss = val_loss_map[i][j]
        cur_normal = val_normal_map[i][j]

        loss_factor = check_grayscale(cur_loss, cur_normal)
        if left_threshold < loss_factor < right_threshold:
        # if loss_factor < -0.25:
            loss_factor_0 = 1
        elif loss_factor == 255:
            loss_factor_0 = -255
        else:
            loss_factor_0 = 0
        val_loss_factor_map[i][j] = loss_factor_0
# +---------------------------------------------------------+

train_df = pd.DataFrame()
b = []
for i in range(0, args.epochs):
    b.append(i)
train_df['epoch'] = b
train_loss = []
val_loss = []
val_acc = []
adjusted_acc = []
other_mIoU = []

best_pred = 0.0
best_pred_epoch = 0
time_epoch = time.time()
for epoch in range(args.epochs):
    optimizer_lr_scheduler.step(epoch)
    print('Epoch:{}'.format(epoch))
    total_loss = train(epoch, optimizer, train_loader, loss_factor_map)
    # if (epoch+1) % 2 ==0:
    if (epoch+1) % 1 == 0:
        best_pred, best_pred_epoch, v_loss, v_acc, Adjusted_mIoU, Other_mIoU = validation(epoch, best_pred, best_pred_epoch, loss_factor_map, val_loss_factor_map, args.save_path)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        adjusted_acc.append(Adjusted_mIoU)
        other_mIoU.append(Other_mIoU)
    else:
        val_loss.append('')
        val_acc.append('')
        adjusted_acc.append('')
        other_mIoU.append('')
    train_loss.append(total_loss)
    print('The {} epoch running time is {} minutes'.format(epoch, (time.time() - time_epoch)/60))
    time_epoch = time.time()

train_df['train_loss'] = train_loss
train_df['val_loss'] = val_loss
train_df['val_acc'] = val_acc
train_df['adjusted_mIoU'] = adjusted_acc
train_df['other_mIoU'] = other_mIoU

cur_pkl_save_path = args.save_path + '/Top_100_epoch.pkl'
train_df.to_pickle(cur_pkl_save_path)

print('The final running time is {} minutes'.format((time.time() - time_00)/60))
