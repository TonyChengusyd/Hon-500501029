import torch
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
from torchvision import transforms
from data.utils import decode_seg_map_sequence
from torchvision.utils import save_image
import numpy as np
import json
import cv2
import os
import time

from torchvision import transforms


def Normalize(x):
    x = np.array(x).astype(np.float32)
    x /=255.0
    x-=(0.485,0.456,0.406)
    x /=(0.229,0.224,0.225)
    return x

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


with open("./uwhvf/alldata.json") as fin:
    dat = json.loads(fin.read())
    
value_map = np.array(dat['data']['18']['R'][0]['hvf'])
loss_map = np.array(dat['data']['18']['R'][0]['td'])
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

loss_map_height = loss_map.shape[0]
loss_map_width = loss_map.shape[1]
loss_factor_map = np.empty((loss_map_height, loss_map_width))
# print(loss_factor_map.shape)
for m in range(loss_map_height):
    for n in range(loss_map_width):
        cur_loss = loss_map[m][n]
        cur_normal = normal_map[m][n]

        loss_factor = check_grayscale(cur_loss, cur_normal)
        if loss_factor < -0.2:
            loss_factor_0 = 1
        else:
            loss_factor_0 = 0
        loss_factor_map[m][n] = loss_factor_0
# +---------------------------------------------------------+


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
danet = torch.load('./training_data/TANet_QKV_has_VFA/TANet_best_epoch_99_mIOU_0.5782.pth',weights_only = False)
danet = danet.to(device)
danet = danet.eval()

baseline = torch.load('./training_data/DSCAMSFF_TANet_QKV_has_VFA/dscamsff_tanet_best_epoch_87_mIOU_0.5841.pth',weights_only = False)
baseline = baseline.to(device)
baseline = baseline.eval()

folder_path = "./images_original/2"
files = sorted(os.listdir(folder_path))
count = 1
time_0 = time.time()
cur_time = time.time()
for file in files:
    print(file)
    pic_path = os.path.join(folder_path, file)

    # pic_path = './s1.png'
    # pic = scipy.misc.imread(pic_path,mode='RGB')
    pic = Image.open(pic_path).convert('RGB')

    pic = pic.resize((1024,512),Image.BILINEAR)

    pic = np.array(pic)
    pic = Normalize(pic)

    pic = np.transpose(pic,(2,0,1))
    pic = torch.from_numpy(pic.copy()).float()
    pic = pic.unsqueeze(0)

    pic = pic.to(device)

    out = danet(pic, loss_factor_map)
    out_all = out[0]
    out_p = out[1]
    out_c = out[2]
    out = out_all.data.cpu().numpy()
    out = np.argmax(out,axis=1)
    pre_danet = decode_seg_map_sequence(out, plot=False)
    save_image(pre_danet, r'./images_pred_danet/2/danet_{}.png'.format(count))

    out_baseline = baseline(pic)
    out_all_baseline = out_baseline[0]
    out_p_baseline = out_baseline[1]
    out_c_baseline = out_baseline[2]
    out_baseline = out_all_baseline.data.cpu().numpy()
    out_baseline = np.argmax(out_baseline,axis=1)
    pre_baseline = decode_seg_map_sequence(out_baseline, plot=False)
    save_image(pre_baseline, r'./images_pred_baseline/2/baseline_{}.png'.format(count))

    print('Pic NO.{}, pic shape:{}'.format(count, pic.size))
    print('Current Pic uses time: {} seconds'.format(time.time() - cur_time))

    count = count + 1
    cur_time = time.time()

print('Total time: {} minutes'.format(round((time.time() - time_0)/60), 4))
