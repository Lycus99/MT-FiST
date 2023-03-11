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


# time_now = time.strftime("%m%d-%H%M", time.localtime())
# best_model_dir = os.path.join('./best_model', time_now)
# Path(best_model_dir).mkdir(parents=True, exist_ok=True)


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='MCNet')
parser.add_argument('--model', default='all',type=str)
parser.add_argument('--task', default=[0], type=int) # 0 tool, 1 verb, 2 target, 3 triplet

parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int)
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=640, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=640, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=20, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=16, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
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

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

# map_dict_url = './dict/maps.txt'
# maps_dict    = np.genfromtxt(map_dict_url, dtype=int, comments='#', delimiter=',', skip_header=0)

# IVT, I, V, T = [], [], [], []
# with open('./dict/maps.txt', 'r') as f:
#     for i, line in enumerate(f.readlines()):
#         if i==0: continue
#         v1, v2, v3, v4 = line.strip().split(',')[:4]
#         IVT.append(int(v1))
#         I.append(int(v2))
#         V.append(int(v3))
#         T.append(int(v4))
# IVT, I, V, T = np.array(IVT), np.array(I), np.array(V), np.array(T)

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

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
    # path = 'logs/004/'
    # writer = SummaryWriter('runs/logs/001')
    record_np = np.zeros([epochs+1, 9])

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

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

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_val_all = len(val_idx)

    # pdb.set_trace()

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False,
        drop_last=True
    )

    model_weight = './best_model/1213-1503/007_epoch_5_length_6_opt_1_mulopt_1_flip_1_crop_1_batch_360.pth'
    file_name = '1213-1503'
    os.makedirs('./results/'+file_name)
    print(model_weight)

    model = model_provider('mc_lstm')
    model.load_state_dict(torch.load(model_weight))

    if use_gpu:
        model = DataParallel(model)
        # model.to(device)
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss(size_average=False)

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
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.max.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_1.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_2.parameters(), 'lr': learning_rate},
                {'params': model.module.classifier_3.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_ap_1, best_val_ap_2, best_val_ap_3, best_val_ap_4 = 0.0, 0.0, 0.0, 0.0
    best_epoch_1, best_epoch_2, best_epoch_3, best_epoch_4 = 0, 0, 0, 0

    # cor_train_ap_1, cor_train_ap_2, cor_train_ap_3, cor_train_ap_4 = 0.0

    l1_save = np.zeros((0, 6))
    l2_save = np.zeros((0, 10))
    l3_save = np.zeros((0, 15))
    l4_save = np.zeros((0, 100))

    p1_save = np.zeros((0, 6))
    p2_save = np.zeros((0, 10))
    p3_save = np.zeros((0, 15))
    p4_save = np.zeros((0, 100))

    for epoch in range(1):
        # Sets the module in evaluation mode.
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        val_progress = 0

        val_all_preds_1, val_all_preds_2, val_all_preds_3, val_all_preds_4  = [], [], [], []
        val_all_labels_1, val_all_labels_2, val_all_labels_3, val_all_labels_4 = [], [], [], []

        with torch.no_grad():
            for j, data in enumerate(val_loader):
                
                inputs, labels_1 = Variable(data[0].cuda()), Variable(data[1].cuda())
                labels_2, labels_3, labels_4 = Variable(data[2].cuda()), Variable(data[3].cuda()), Variable(data[4].cuda())

                outputs, loss = model.forward(inputs, labels_1, labels_2, labels_3, labels_4, sequence_length)
                outputs_1, outputs_2, outputs_3, outputs_4 = outputs

                labels_1, labels_2, labels_3, labels_4 = labels_1.data.float(), labels_2.data.float(), labels_3.data.float(), labels_4.data.float()

                # pdb.set_trace()
                labels_1 = np.array(labels_1.cpu())
                labels_2 = np.array(labels_2.cpu())
                labels_3 = np.array(labels_3.cpu())
                labels_4 = np.array(labels_4.cpu())

                sig_out1, sig_out2, sig_out3 = sig_f(outputs_1.data), sig_f(outputs_2.data), sig_f(outputs_3.data)
                sig_out4 = sig_f(outputs_4.data)

                out1 = np.array(sig_out1.cpu().data)
                out2 = np.array(sig_out2.cpu().data)
                out3 = np.array(sig_out3.cpu().data)
                out4 = np.array(sig_out4.cpu().data)
                #
                l1_save = np.vstack((l1_save, labels_1))
                l2_save = np.vstack((l2_save, labels_2))
                l3_save = np.vstack((l3_save, labels_3))
                l4_save = np.vstack((l4_save, labels_4))
                #
                p1_save, p2_save, p3_save, p4_save = np.vstack((p1_save, out1)), np.vstack((p2_save, out2)), np.vstack((p3_save, out3)), np.vstack((p4_save, out4))



                # preds_1 = torch.from_numpy(np.array(sig_out1.cpu() > 0.5, dtype=int))
                # preds_2 = torch.from_numpy(np.array(sig_out2.cpu() > 0.5, dtype=int))
                # preds_3 = torch.from_numpy(np.array(sig_out3.cpu() > 0.5, dtype=int))
                # preds_4 = torch.from_numpy(np.array(sig_out4.cpu() > 0.5, dtype=int))
                #
                # preds_1, preds_2, preds_3 = preds_1.float(), preds_2.float(), preds_3.float()
                # preds_4 = preds_4.float()

                # if j != 0:
                #     val_total_labels_1, val_total_labels_2, val_total_labels_3 = torch.cat((val_total_labels_1, labels_1.data.cpu()), dim=0), torch.cat((val_total_labels_2, labels_2.data.cpu()), dim=0), torch.cat((val_total_labels_3, labels_3.data.cpu()), dim=0)
                #     val_total_labels_4 = torch.cat((val_total_labels_4, labels_4.data.cpu()), dim=0)
                #     val_total_preds_1, val_total_preds_2, val_total_preds_3 = torch.cat((val_total_preds_1, preds_1), dim=0), torch.cat((val_total_preds_2, preds_2), dim=0), torch.cat((val_total_preds_3, preds_3), dim=0)
                #     val_total_preds_4 = torch.cat((val_total_preds_4, preds_4), dim=0)
                # else:
                #     val_total_labels_1, val_total_labels_2, val_total_labels_3 = labels_1.data.cpu(), labels_2.data.cpu(), labels_3.data.cpu()
                #     val_total_labels_4 = labels_4.data.cpu()
                #     val_total_preds_1, val_total_preds_2, val_total_preds_3 = preds_1, preds_2, preds_3
                #     val_total_preds_4 = preds_4
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
                
                val_progress += 1
                if val_progress*val_batch_size >= num_val_all:
                    percent = 100.0
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', num_val_all, num_val_all), end='\n')
                else:
                    percent = round(val_progress*val_batch_size / num_val_all * 100, 2)
                    print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress*val_batch_size, num_val_all), end='\r')

        np.save('./results/' + file_name + '/p1.npy', p1_save)
        np.save('./results/' + file_name + '/p2.npy', p2_save)
        np.save('./results/' + file_name + '/p3.npy', p3_save)
        np.save('./results/' + file_name + '/p4.npy', p4_save)

        np.save('./results/' + file_name + '/l1.npy', l1_save)
        np.save('./results/' + file_name + '/l2.npy', l2_save)
        np.save('./results/' + file_name + '/l3.npy', l3_save)
        np.save('./results/' + file_name + '/l4.npy', l4_save)
        #
        # pdb.set_trace()

        # val_total_labels_1 = val_total_labels_1[sequence_length - 1::sequence_length]
        # val_total_labels_2 = val_total_labels_2[sequence_length - 1::sequence_length]
        # val_total_labels_3 = val_total_labels_3[sequence_length - 1::sequence_length]
        # val_total_labels_4 = val_total_labels_4[sequence_length - 1::sequence_length]
        #
        # val_total_preds_1 = val_total_preds_1[sequence_length - 1::sequence_length]
        # val_total_preds_2 = val_total_preds_2[sequence_length - 1::sequence_length]
        # val_total_preds_3 = val_total_preds_3[sequence_length - 1::sequence_length]
        # val_total_preds_4 = val_total_preds_4[sequence_length - 1::sequence_length]
        recognize = ivt_metrics.Recognition(num_class=100)
        recognize.reset()
        recognize.update(torch.tensor(l4_save[sequence_length-1::sequence_length, :]), torch.tensor(p4_save[sequence_length-1::sequence_length, :]))

        results_i = recognize.compute_AP('i')
        results_v = recognize.compute_AP('v')
        results_t = recognize.compute_AP('t')
        val_ap_i = results_i["mAP"]
        val_ap_v = results_v["mAP"]
        val_ap_t = results_t["mAP"]
        iv = recognize.compute_AP('iv')['mAP']
        it = recognize.compute_AP('it')['mAP']

        # results_iv = recognize.compute_AP('iv')
        # results_it = recognize.compute_AP('it')
        # val_ap_iv = results_iv["mAP"]
        # val_ap_it = results_it["mAP"]

        results_ivt = recognize.compute_AP('ivt')
        val_ap_ivt = results_ivt["mAP"]

        val_ap_each_1, val_ap_1 = _compute_AP(gt_labels=l1_save[sequence_length-1::sequence_length,:], pd_probs=p1_save[sequence_length-1::sequence_length,:], valid=None)
        val_ap_each_2, val_ap_2 = _compute_AP(gt_labels=l2_save[sequence_length-1::sequence_length,:], pd_probs=p2_save[sequence_length-1::sequence_length,:], valid=None)
        val_ap_each_3, val_ap_3 = _compute_AP(gt_labels=l3_save[sequence_length-1::sequence_length,:], pd_probs=p3_save[sequence_length-1::sequence_length,:], valid=None)
        val_ap_each_4, val_ap_4 = _compute_AP(gt_labels=l4_save[sequence_length-1::sequence_length,:], pd_probs=p4_save[sequence_length-1::sequence_length,:], valid=None)

        print(' valid ap1 (I/V/T/IVT): {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(val_ap_1, val_ap_2, val_ap_3, val_ap_4))

        print(' valid ap rdv (I/V/T/IV/IT/IVT): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(val_ap_i, val_ap_v, val_ap_t,iv,it,val_ap_ivt))

        # val_elapsed_time = time.time() - val_start_time
        # val_average_loss = val_loss / num_val_all
        #
        # val_all_preds_1 = np.array(val_all_preds_1)
        # val_all_preds_2 = np.array(val_all_preds_2)
        # val_all_preds_3 = np.array(val_all_preds_3)
        # val_all_preds_4 = np.array(val_all_preds_4)
        # val_all_labels_1 = np.array(val_all_labels_1)
        # val_all_labels_2 = np.array(val_all_labels_2)
        # val_all_labels_3 = np.array(val_all_labels_3)
        # val_all_labels_4 = np.array(val_all_labels_4)
        #
        # val_precision_each_1 = metrics.precision_score(val_total_labels_1, val_total_preds_1, average=None)
        # # val_precision_each_1 = metrics.precision_score(val_all_labels_1,val_all_preds_1, average=None)
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


        # print('epoch: {:4d}'
        #       ' valid in: {:2.0f}m{:2.0f}s'
        #       ' valid ap (T/V/T/triplet): {:.4f}/{:.4f}/{:.4f}/{:.4f}'
        #       .format(epoch,
        #               val_elapsed_time // 60, val_elapsed_time % 60,
        #               val_ap_1, val_ap_2, val_ap_3, val_ap_4
        #               ))
    #     # pdb.set_trace()
    #
    #
    # print(val_precision_1, val_precision_2, val_precision_3, val_precision_4)
    # print(val_recall_1, val_recall_2, val_recall_3, val_recall_4)


def main():
    # val: 74,75,78,79,80
    train_dataset, train_num_each, val_dataset, val_num_each = get_data('preprocess/'+pkl)
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()