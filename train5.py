import pdb

from PIL import Image
import time
import pickle
import numpy as np
import argparse
import copy
import os
import numbers
import random
from PIL import Image, ImageOps
from sklearn import metrics
from sklearn.metrics import average_precision_score
import cv2
import warnings
import time
from pathlib import Path
import ivtmetrics.recognition as ivt_metrics


time_now = time.strftime("%m%d-%H%M", time.localtime())
best_model_dir = os.path.join('./best_model', time_now)
Path(best_model_dir).mkdir(parents=True, exist_ok=True)
print(time_now)

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='MCNet')
parser.add_argument('--model', default='all',type=str)
parser.add_argument('--task', default=[0], type=int) # 0 tool, 1 verb, 2 target, 3 triplet

parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int)
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=320, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=320, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=10, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-3, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('-p', '--pkl', default='tr36_val9.pkl', type=str, help='pkl file for run')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('-fz', '--freeze', default=False, type=bool, help='freeze net, default True')


args = parser.parse_args()

model_used = args.model
task_used = args.task
gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov

pkl = args.pkl
print(pkl)

sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma

freeze_net = args.freeze

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
print(args)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from models import model_provider
from my_pooling import my_MaxPool2d,my_AvgPool2d
from torchvision.transforms import Lambda
import logging


num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

map_dict_url = './dict/maps.txt'
maps_dict = np.genfromtxt(map_dict_url, dtype=int, comments='#', delimiter=',', skip_header=0)

IVT, I, V, T = [], [], [], []
with open('./dict/maps.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i==0: continue
        v1, v2, v3, v4 = line.strip().split(',')[:4]
        IVT.append(int(v1))
        I.append(int(v2))
        V.append(int(v3))
        T.append(int(v4))
IVT, I, V, T = np.array(IVT), np.array(I), np.array(V), np.array(T)

recognize = ivt_metrics.Recognition(num_class=100)

m1 = 0.005
m2 = 2 * m1
print('m1: ', m1, 'm2: ', m2)

model_name = 'all4'
print(model_name)


def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)

    pth = './best_model/' + time_now + '/log.txt'

    fHandler = logging.FileHandler(pth, mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)
    return logger

logger = getLogger()
logger.info("m1: {}, m2: {}".format(m1, m2))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))

class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotation(object):
    def __init__(self,degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees,self.degrees)
        return TF.rotate(img, angle)

class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img,brightness_factor)
        img_ = TF.adjust_contrast(img_,contrast_factor)
        img_ = TF.adjust_saturation(img_,saturation_factor)
        img_ = TF.adjust_hue(img_,hue_factor)
        
        return img_


class TripletDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, 0:-125].squeeze()
        self.file_labels_2 = file_labels[:, 6:-115].squeeze()
        self.file_labels_3 = file_labels[:, 16:-100].squeeze()
        self.file_labels_4 = file_labels[:, 31:].squeeze()
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        labels_3 = self.file_labels_3[index]
        labels_4 = self.file_labels_4[index]
        # print(img_names)
        imgs = self.loader(img_names)
        # print(img_names)
        # imgs = cv2.imread(img_names)
        # imgs = Image.fromarray(cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_1, labels_2, labels_3, labels_4

    def __len__(self):
        return len(self.file_paths)


def get_useful_start_idx(sequence_length, list_each_length):
    # 0-10; 1-11; ... ; 1724-1734
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    train_labels = train_test_paths_labels[2]
    val_labels = train_test_paths_labels[3]
    train_num_each = train_test_paths_labels[4]
    val_num_each = train_test_paths_labels[5]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.44893518,0.3226702,0.34424525],[0.22357443,0.18503027,0.1900281])(crop) for crop in crops]))
        ])

    train_dataset = TripletDataset(train_paths, train_labels, train_transforms)
    val_dataset = TripletDataset(val_paths, val_labels, test_transforms)
    # test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


sig_f = nn.Sigmoid()


def _compute_AP(gt_labels, pd_probs, valid=None):
    """ Compute the average precision (AP) of each of the 100 considered triplets.
        Args:
            gt_labels: 1D (batch of) vector[N] of integer values 0's or 1's for the groundtruth labels.
            pd_probs:  1D (batch of) vector[N] of float values [0,1] for the predicted labels.
        Returns:
            results:   1D vector[N] of AP for each class 
    """
    # gt_instances  = np.sum(gt_labels, axis=0)
    # pd_instances  = np.sum(pd_probs, axis=0)
    # print(gt_labels, pd_probs)
    # print(gt_labels.shape, pd_probs.shape)
    # print(pd_probs.shape)
    computed_ap   = average_precision_score(gt_labels, pd_probs, average=None) 
    actual_ap     = []
    empty = 0.0
    num_classes   = np.shape(gt_labels)[-1]
    for k in range(num_classes):
        # if ((gt_instances[k] != 0) or (pd_instances[k] != 0)) and not np.isnan(computed_ap[k]):
        if not np.isnan(computed_ap[k]):
            actual_ap.append(computed_ap[k])
        else:
            # actual_ap.append("n/a")
            actual_ap.append(empty)
    # print(actual_ap)
    mAP = np.mean([float(a) for a in actual_ap if a != 0.0 ]) 
    return actual_ap, mAP
    # return actual_ap


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    # TensorBoard
    # if not os.path.exists('logs/004'):
    #     os.mkdir('logs/004')


    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    # len(train_useful_start_idx) = len(train_num_each) - sequence_length * 40
    num_train_we_use = len(train_useful_start_idx)
    num_val_we_use = len(val_useful_start_idx)
    # num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    # num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # num_train_we_use = 8000
    # num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    # np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)


    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False,
        drop_last=True
    )


    model = model_provider(model_name)


    if use_gpu:
        model = DataParallel(model)
        # model.to(device)
        model = model.cuda()

    # criterion = nn.BCEWithLogitsLoss(size_average=False)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = None
    exp_lr_scheduler = None

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.lstm.parameters(), 'lr': learning_rate},
                # {'params': model.module.fc.parameters(), 'lr': learning_rate},
                {'params': model.module.fc1.parameters(), 'lr': learning_rate},
                # {'params': model.module.fc2.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.AdamW([
                {'params': model.module.share.parameters()},
                {'params': model.module.max.parameters(), 'lr': learning_rate},
                {'params': model.module.features2fixed_1.parameters(), 'lr': learning_rate},
                {'params': model.module.features2fixed_2.parameters(), 'lr': learning_rate},
                {'params': model.module.features2fixed_3.parameters(), 'lr': learning_rate},
                {'params': model.module.features2fixed_4.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_1.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_2.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_3.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_4.parameters(), 'lr': learning_rate},
                {'params': model.module.lstm1.parameters(), 'lr': learning_rate},
                {'params': model.module.lstm2.parameters(), 'lr': learning_rate},
                {'params': model.module.fc_h.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    # best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_ap_1, best_val_ap_2, best_val_ap_3, best_val_ap_4 = 0.0, 0.0, 0.0, 0.0
    best_epoch_1, best_epoch_2, best_epoch_3, best_epoch_4 = 0, 0, 0, 0

    # cor_train_ap_1, cor_train_ap_2, cor_train_ap_3, cor_train_ap_4 = 0.0

    for epoch in range(epochs):
        # np.random.seed(epoch)
        loss_bce1, loss_bce2, loss_bce3, loss_bce4 = 0.0, 0.0, 0.0, 0.0
        loss_mc1 = 0.0
        loss_mc2 = 0.0

        recognize.reset()

        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=workers,
            pin_memory=False,
            drop_last=True
        )

        # Sets the module in training mode.
        model.train()
        train_loss = 0.0
        
        batch_progress = 0.0
        running_loss = 0.0
        train_start_time = time.time()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_1 = Variable(data[0].cuda()), Variable(data[1].cuda())
                labels_2, labels_3, labels_4 = Variable(data[2].cuda()), Variable(data[3].cuda()), Variable(data[4].cuda())

            # pdb.set_trace()
            outputs, loss, MC_loss = model.forward(inputs, labels_1, labels_2, labels_3, labels_4, sequence_length)
            outputs_1, outputs_2, outputs_3, outputs_4 = outputs
            total_mc_loss_0 = torch.mean(MC_loss[0][0]) + torch.mean(MC_loss[1][0]) + torch.mean(MC_loss[2][0]) + torch.mean(MC_loss[3][0])
            total_mc_loss_1 = torch.mean(MC_loss[0][1]) + torch.mean(MC_loss[1][1]) + torch.mean(MC_loss[2][1]) + torch.mean(MC_loss[3][1])

            labels_1, labels_2, labels_3, labels_4 = labels_1.data.float(), labels_2.data.float(), labels_3.data.float(), labels_4.data.float()
            loss_1, loss_2, loss_3 = criterion(outputs_1, labels_1), criterion(outputs_2, labels_2), criterion(outputs_3, labels_3)
            loss_4 = criterion(outputs_4, labels_4)
            # print(MC_loss)
            loss = loss_1 + loss_2 + loss_3 + loss_4 + m1 * total_mc_loss_0 + m2 * total_mc_loss_1
            # pdb.set_trace()
            # sig_out1, sig_out2, sig_out3 = sig_f(outputs_1.data), sig_f(outputs_2.data), sig_f(outputs_3.data)
            # sig_out4 = sig_f(outputs_4.data)

            # preds_1, preds_2, preds_3, preds_4 = sig_out1.cpu(), sig_out2.cpu(), sig_out3.cpu(), sig_out4.cpu()

            # preds_1 = torch.from_numpy(np.array(sig_out1.cpu() > 0.5, dtype=int))
            # preds_2 = torch.from_numpy(np.array(sig_out2.cpu() > 0.5, dtype=int))
            # preds_3 = torch.from_numpy(np.array(sig_out3.cpu() > 0.5, dtype=int))
            # preds_4 = torch.from_numpy(np.array(sig_out4.cpu() > 0.5, dtype=int))
            # pdb.set_trace()
            # preds_1, preds_2, preds_3 = (sig_out1.cpu() > 0.5).mul_(1), (sig_out2.cpu() > 0.5).mul_(1), (sig_out3.cpu() > 0.5).mul_(1)
            # preds_4 = (sig_out4.cpu() > 0.5).mul_(1)

            # pdb.set_trace()
            # recognize.update(labels_4.cpu(), preds_4)


            loss.backward()
            optimizer.step()

            loss_bce1 += loss_1.data.item()/len(train_loader)
            loss_bce2 += loss_2.data.item()/len(train_loader)
            loss_bce3 += loss_3.data.item()/len(train_loader)
            loss_bce4 += loss_4.data.item()/len(train_loader)
            loss_mc1 += (m1 * total_mc_loss_0).data.item()/len(train_loader)
            loss_mc2 += (m1 * total_mc_loss_1).data.item()/len(train_loader)
            # running_loss += loss.data.item()
            # train_loss += loss.data.item()

            # if i != 0:
            #     # pdb.set_trace()
            #     total_labels_1, total_labels_2, total_labels_3 = torch.cat((total_labels_1, labels_1.data.cpu()), dim=0), torch.cat((total_labels_2, labels_2.data.cpu()), dim=0), torch.cat((total_labels_3, labels_3.data.cpu()), dim=0)
            #     total_labels_4 = torch.cat((total_labels_4, labels_4.data.cpu()), dim=0)
            #     total_preds_1, total_preds_2, total_preds_3 = torch.cat((total_preds_1, preds_1), dim=0), torch.cat((total_preds_2, preds_2), dim=0), torch.cat((total_preds_3, preds_3), dim=0)
            #     total_preds_4 = torch.cat((total_preds_4, preds_4), dim=0)
            # else:
            #     total_labels_1, total_labels_2, total_labels_3, total_labels_4 = labels_1.data.cpu(), labels_2.data.cpu(), labels_3.data.cpu(), labels_4.data.cpu()
            #     total_preds_1, total_preds_2, total_preds_3 = preds_1, preds_2, preds_3
            #     total_preds_4 = preds_4

            # if i % 500 == 499:
            # if i % 50000 == 49999:
            #     # ...log the running loss
            #     batch_iters = epoch * num_train_all/sequence_length + i*train_batch_size/sequence_length
            #     # writer.add_scalar('training loss',
            #     #                   running_loss / (train_batch_size*50000) / 7,
            #     #                   batch_iters)
            #
            #     # ...log the val acc loss
            #     # val_loss = valMinibatch(val_loader, model)
            #     # writer.add_scalar('validation loss miniBatch tool',
            #     #                   float(val_loss) / float(num_val_all) / 7,
            #     #                   batch_iters)
            #     val_loss = 0.0
            #
            #     running_loss = 0.0
            #
            # if (i+1)*train_batch_size >= num_train_all:
            #     running_loss = 0.0

            batch_progress += 1
            if batch_progress*train_batch_size >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            else:
                percent = round(batch_progress*train_batch_size / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress*train_batch_size, num_train_all), end='\r')

        # pdb.set_trace()
        # total_labels_1 = total_labels_1[sequence_length-1::sequence_length]
        # total_labels_2 = total_labels_2[sequence_length-1::sequence_length]
        # total_labels_3 = total_labels_3[sequence_length-1::sequence_length]
        # total_labels_4 = total_labels_4[sequence_length-1::sequence_length]
        #
        # total_preds_1 = total_preds_1[sequence_length-1::sequence_length]
        # total_preds_2 = total_preds_2[sequence_length - 1::sequence_length]
        # total_preds_3 = total_preds_3[sequence_length - 1::sequence_length]
        # total_preds_4 = total_preds_4[sequence_length - 1::sequence_length]
        #
        # ap_each_1, ap_1 = _compute_AP(gt_labels=total_labels_1, pd_probs=total_preds_1, valid=None)
        # ap_each_2, ap_2 = _compute_AP(gt_labels=total_labels_2, pd_probs=total_preds_2, valid=None)
        # ap_each_3, ap_3 = _compute_AP(gt_labels=total_labels_3, pd_probs=total_preds_3, valid=None)
        # ap_each_4, ap_4 = _compute_AP(gt_labels=total_labels_4, pd_probs=total_preds_4, valid=None)
        #
        # recognize.update(total_labels_4, total_preds_4)
        # results_iv = recognize.compute_AP('iv')
        # results_it = recognize.compute_AP('it')
        # ap_iv = results_iv["mAP"]
        # ap_it = results_it["mAP"]

        train_elapsed_time = time.time() - train_start_time

        # Sets the module in evaluation mode.
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        val_progress = 0

        # val_all_preds_1, val_all_preds_2, val_all_preds_3, val_all_preds_4  = [], [], [], []
        # val_all_labels_1, val_all_labels_2, val_all_labels_3, val_all_labels_4 = [], [], [], []

        # val_ap_1, val_ap_2, val_ap_3, val_ap_4 = 0.0, 0.0, 0.0, 0.0

        # recognize.reset_global()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                
                inputs, labels_1 = Variable(data[0].cuda()), Variable(data[1].cuda())
                labels_2, labels_3, labels_4 = Variable(data[2].cuda()), Variable(data[3].cuda()), Variable(data[4].cuda())

                outputs, loss = model.forward(inputs, labels_1, labels_2, labels_3, labels_4, sequence_length)
                outputs_1, outputs_2, outputs_3, outputs_4 = outputs

                labels_1, labels_2, labels_3, labels_4 = labels_1.data.cpu().float(), labels_2.data.cpu().float(), labels_3.data.cpu().float(), labels_4.data.cpu().float()
                # labels_1, labels_2, labels_3, labels_4 = labels_1.data.float(), labels_2.data.float(), labels_3.data.float(), labels_4.data.float()

                sig_out1, sig_out2, sig_out3 = sig_f(outputs_1.data), sig_f(outputs_2.data), sig_f(outputs_3.data)
                sig_out4 = sig_f(outputs_4.data)

                preds_1, preds_2, preds_3, preds_4 = sig_out1.cpu(), sig_out2.cpu(), sig_out3.cpu(), sig_out4.cpu()
                preds_1, preds_2, preds_3, preds_4 = preds_1.float(), preds_2.float(), preds_3.float(), preds_4.float()

                labels_1 = labels_1[sequence_length - 1::sequence_length]
                labels_2 = labels_2[sequence_length - 1::sequence_length]
                labels_3 = labels_3[sequence_length - 1::sequence_length]
                labels_4 = labels_4[sequence_length - 1::sequence_length]

                preds_1 = preds_1[sequence_length - 1::sequence_length]
                preds_2 = preds_2[sequence_length - 1::sequence_length]
                preds_3 = preds_3[sequence_length - 1::sequence_length]
                preds_4 = preds_4[sequence_length - 1::sequence_length]


                # val_ap_1 += _compute_AP(gt_labels=labels_1, pd_probs=preds_1, valid=None)[1]/len(val_loader)
                # val_ap_2 += _compute_AP(gt_labels=labels_2, pd_probs=preds_2, valid=None)[1]/len(val_loader)
                # val_ap_3 += _compute_AP(gt_labels=labels_3, pd_probs=preds_3, valid=None)[1]/len(val_loader)
                # val_ap_4 += _compute_AP(gt_labels=labels_4, pd_probs=preds_4, valid=None)[1]/len(val_loader)

                # pdb.set_trace()

                if j != 0:
                    val_total_labels_1, val_total_labels_2, val_total_labels_3 = torch.cat((val_total_labels_1, labels_1.data.cpu()), dim=0), torch.cat((val_total_labels_2, labels_2.data.cpu()), dim=0), torch.cat((val_total_labels_3, labels_3.data.cpu()), dim=0)
                    val_total_labels_4 = torch.cat((val_total_labels_4, labels_4.data.cpu()), dim=0)
                    val_total_preds_1, val_total_preds_2, val_total_preds_3 = torch.cat((val_total_preds_1, preds_1), dim=0), torch.cat((val_total_preds_2, preds_2), dim=0), torch.cat((val_total_preds_3, preds_3), dim=0)
                    val_total_preds_4 = torch.cat((val_total_preds_4, preds_4), dim=0)
                else:
                    val_total_labels_1, val_total_labels_2, val_total_labels_3 = labels_1.data.cpu(), labels_2.data.cpu(), labels_3.data.cpu()
                    val_total_labels_4 = labels_4.data.cpu()
                    val_total_preds_1, val_total_preds_2, val_total_preds_3 = preds_1, preds_2, preds_3
                    val_total_preds_4 = preds_4
                #
                # for i in range(len(preds_1)):
                #     val_all_preds_1.append(list(preds_1.data.cpu()[i]))
                # for i in range(len(labels_1)):
                #     val_all_labels_1.append(list(labels_1.data.cpu()[i]))
                #
                # for i in range(len(preds_2)):
                #     val_all_preds_2.append(list(preds_2.data.cpu()[i]))
                # for i in range(len(labels_2)):
                #     val_all_labels_2.append(list(labels_2.data.cpu()[i]))
                #
                # for i in range(len(preds_3)):
                #     val_all_preds_3.append(list(preds_3.data.cpu()[i]))
                # for i in range(len(labels_3)):
                #     val_all_labels_3.append(list(labels_3.data.cpu()[i]))
                #
                # for i in range(len(preds_4)):
                #     val_all_preds_4.append(list(preds_4.data.cpu()[i]))
                # for i in range(len(labels_4)):
                #     val_all_labels_4.append(list(labels_4.data.cpu()[i]))
                #
                val_progress += 1
                if val_progress*val_batch_size >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        # val_total_labels_1 = val_total_labels_1[sequence_length - 1::sequence_length]
        # val_total_labels_2 = val_total_labels_2[sequence_length - 1::sequence_length]
        # val_total_labels_3 = val_total_labels_3[sequence_length - 1::sequence_length]
        # val_total_labels_4 = val_total_labels_4[sequence_length - 1::sequence_length]
        #
        # val_total_preds_1 = val_total_preds_1[sequence_length - 1::sequence_length]
        # val_total_preds_2 = val_total_preds_2[sequence_length - 1::sequence_length]
        # val_total_preds_3 = val_total_preds_3[sequence_length - 1::sequence_length]
        # val_total_preds_4 = val_total_preds_4[sequence_length - 1::sequence_length]
        # #
        # val_ap_each_1, val_ap_1 = _compute_AP(gt_labels=val_total_labels_1, pd_probs=val_total_preds_1, valid=None)
        # val_ap_each_2, val_ap_2 = _compute_AP(gt_labels=val_total_labels_2, pd_probs=val_total_preds_2, valid=None)
        # val_ap_each_3, val_ap_3 = _compute_AP(gt_labels=val_total_labels_3, pd_probs=val_total_preds_3, valid=None)
        # val_ap_each_4, val_ap_4 = _compute_AP(gt_labels=val_total_labels_4, pd_probs=val_total_preds_4, valid=None)

        recognize.update(val_total_labels_4, val_total_preds_4)
        val_ap_i = recognize.compute_AP('i')["mAP"]
        val_ap_v = recognize.compute_AP('v')["mAP"]
        val_ap_t = recognize.compute_AP('t')["mAP"]
        val_ap_iv = recognize.compute_AP('iv')["mAP"]
        val_ap_it = recognize.compute_AP('it')["mAP"]
        val_ap_ivt = recognize.compute_AP('ivt')["mAP"]

        val_elapsed_time = time.time() - val_start_time
        # val_average_loss = val_loss / num_val_all

        # val_all_preds_1 = np.array(val_all_preds_1)
        # val_all_preds_2 = np.array(val_all_preds_2)
        # val_all_preds_3 = np.array(val_all_preds_3)
        # val_all_preds_4 = np.array(val_all_preds_4)
        # val_all_labels_1 = np.array(val_all_labels_1)
        # val_all_labels_2 = np.array(val_all_labels_2)
        # val_all_labels_3 = np.array(val_all_labels_3)
        # val_all_labels_4 = np.array(val_all_labels_4)

        # val_precision_each_1 = metrics.precision_score(val_all_labels_1,val_all_preds_1, average=None)
        # val_recall_each_1 = metrics.recall_score(val_all_labels_1,val_all_preds_1, average=None)
        # val_precision_1 = metrics.precision_score(val_all_labels_1,val_all_preds_1, average="macro")
        # val_recall_1 = metrics.recall_score(val_all_labels_1,val_all_preds_1, average="macro")
        #
        # val_precision_each_2 = metrics.precision_score(val_all_labels_2, val_all_preds_2, average=None)
        # val_recall_each_2 = metrics.recall_score(val_all_labels_2, val_all_preds_2, average=None)
        # val_precision_2 = metrics.precision_score(val_all_labels_2, val_all_preds_2, average='macro')
        # val_recall_2 = metrics.recall_score(val_all_labels_2, val_all_preds_2, average='macro')
        #
        # val_precision_each_3 = metrics.precision_score(val_all_labels_3, val_all_preds_3, average=None)
        # val_recall_each_3 = metrics.recall_score(val_all_labels_3, val_all_preds_3, average=None)
        # val_precision_3 = metrics.precision_score(val_all_labels_3, val_all_preds_3, average='macro')
        # val_recall_3 = metrics.recall_score(val_all_labels_3, val_all_preds_3, average='macro')
        #
        # val_precision_each_4 = metrics.precision_score(val_all_labels_4, val_all_preds_4, average=None)
        # val_recall_each_4 = metrics.recall_score(val_all_labels_4, val_all_preds_4, average=None)
        # val_precision_4 = metrics.precision_score(val_all_labels_4, val_all_preds_4, average='macro')
        # val_recall_4 = metrics.recall_score(val_all_labels_4, val_all_preds_4, average='macro')


        logger.info('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss(I/V/T/IVT/mc1/mc2): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s' 
              ' valid ap (I/V/T/IV/IT/IVT): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'
              .format(epoch,
                      train_elapsed_time // 60, train_elapsed_time % 60,
                      loss_bce1, loss_bce2, loss_bce3, loss_bce4, loss_mc1, loss_mc2,
                      val_elapsed_time // 60, val_elapsed_time % 60,
                      val_ap_i, val_ap_v, val_ap_t, val_ap_iv, val_ap_it, val_ap_ivt
                      ))

        # print(val_precision_1, val_precision_2, val_precision_3, val_precision_4)
        # print(val_recall_1, val_recall_2, val_recall_3, val_recall_4)

        # save results
        # np.save(path + str(epoch)+ '_pr.npy', val_precision_each_1)
        # np.save(path + str(epoch)+ '_re.npy', val_recall_each_1)
        # np.save(path + str(epoch)+ '_ap.npy', val_ap_each)

        
        # if optimizer_choice == 0:
        #     if sgd_adjust_lr == 0:
        #         exp_lr_scheduler.step()
        #     elif sgd_adjust_lr == 1:
        #         exp_lr_scheduler.step(val_average_loss)

        if val_ap_ivt > best_val_ap_4:
            best_val_ap_4 = val_ap_ivt
            # correspond_train_acc_1 = train_accuracy_1
            # best_val_accuracy_2 = val_accuracy_2
            # correspond_train_acc_2 = train_accuracy_2
            # best_val_accuracy_3 = val_accuracy_3
            # correspond_train_acc_3 = train_accuracy_3
            # best_val_accuracy_4 = val_accuracy_4
            # correspond_train_acc_4 = train_accuracy_4

            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch_4 = epoch
     
            public_name = "007" \
                          + "_epoch_" + str(best_epoch_4) \
                          + "_length_" + str(sequence_length) \
                          + "_opt_" + str(optimizer_choice) \
                          + "_mulopt_" + str(multi_optim) \
                          + "_flip_" + str(use_flip) \
                          + "_crop_" + str(crop_type) \
                          + "_batch_" + str(train_batch_size)
                    #   + "_train_2" + str(save_train_2) \
                    #   + "_train_3" + str(save_train_3) \
                    #   + "_train_4" + str(save_train_4) \
                    #   + "_val2_" + str(save_val_2) \
                    #   + "_val3_" + str(save_val_3) \
                    #   + "_val4_" + str(save_val_4)

            torch.save(best_model_wts, best_model_dir + "/"+public_name+".pth")
        # n.save(public_name+".npy", record_np)
        print("best_epoch", str(best_epoch_4))

        # torch.save(model.module.state_dict(), "./temp_single/007"+str(epoch)+".pth")

    print('best ap: {:.4f} cor epoch: {}'
          .format(best_val_ap_4, best_epoch_4))


def main():
    train_dataset, train_num_each, val_dataset, val_num_each = get_data('preprocess/'+pkl)
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
