import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import DataParallel
from torchvision import models
import random

import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

from my_pooling import my_MaxPool2d,my_AvgPool2d

# net = models.resnet50(pretrained=True)
# net = DataParallel(net)
# net = net.cuda()
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()


def Mask(nb_batch, channels, classes_num):

    foo = [1] * int(channels/2) + [0] *  int(channels/2)
    bar = []
    for i in range(classes_num):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,classes_num*channels,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar


def supervisor(x,targets,height,cnum):
        # classes=6, cnum=340, channels=2040

        # L_div
        branch = x  # x: 10, 2040, 7, 7
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))  # 10, 2040, 49
        branch = F.softmax(branch,2)  # Softmax
        branch = branch.reshape(branch.size(0),branch.size(1), x.size(2), x.size(2))  # 10, 2040, 7, 7
        branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)  # (10, 6, 7, 7) CCMP
        branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))  # 10, 6, 49
        loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch, 2))/cnum  #


        # L_dis
        mask = Mask(x.size(0), cnum, targets.shape[1])
        x, mask = Variable(x.cuda()), Variable(mask.cuda())
        branch_1 = x * mask  # CWA (10,2040,7,7)

        branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1) # (10, 6, 7, 7) CCMP
        branch_1 = nn.AvgPool2d(kernel_size=(height, height))(branch_1)  # GAP  10,6,1,1
        branch_1 = branch_1.view(branch_1.size(0), -1)  # 10,6

        targets = targets.data.float()
        loss_1 = criterion(branch_1, targets)  # Softmax
        
        return [loss_1, loss_2] 


class model_lstm(nn.Module):

    def __init__(self, feature_size=2048):

        super(model_lstm, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.avg_p = resnet.avgpool
        self.lstm1 = nn.LSTM(2040, 512, batch_first=True)
        self.lstm2 = nn.LSTM(2000, 512, batch_first=True)
        self.fc_h = nn.Linear(512, 2040)
        # self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(512, 6), nn.Linear(512, 10), nn.Linear(512, 15), nn.Linear(512, 15)

        init.xavier_normal_(self.lstm1.all_weights[0][0])
        init.xavier_normal_(self.lstm1.all_weights[0][1])
        init.xavier_normal_(self.lstm2.all_weights[0][0])
        init.xavier_normal_(self.lstm2.all_weights[0][1])
        init.xavier_uniform_(self.fc_h.weight)

        self.features2fixed_1 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        self.features2fixed_2 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        self.features2fixed_3 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        self.features2fixed_4 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040*1*1
        self.classifier_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 6),
        )

        self.classifier_2 = nn.Sequential(
            nn.BatchNorm1d(2040*1*1),
            #nn.Dropout(0.5),
            nn.Linear(2040*1*1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 10),
        )

        self.classifier_3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 15),
        )

        self.classifier_4 = nn.Sequential(
            nn.BatchNorm1d(2040*1*1),
            #nn.Dropout(0.5),
            nn.Linear(2040*1*1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 100),
        )

    def forward(self, x, targets_1, targets_2, targets_3, targets_4, seq):
        # pdb.set_trace()
        x = self.share(x)  # bs, 2048, 7, 7
        x1 = self.features2fixed_1(x)  # bs, 2040, 7, 7
        x2 = self.features2fixed_2(x)  # bs, 2000, 7, 7
        x3 = self.features2fixed_3(x)  # bs, 2040, 7, 7
        x4 = self.features2fixed_4(x)  # bs, 2000, 7, 7
        if self.training:
            MC_loss_1 = supervisor(x1,targets_1,height=7,cnum=340)  # c1=6, 340*6=2040
            MC_loss_2 = supervisor(x2,targets_2,height=7,cnum=200)  # c2=10, 200*10=2000
            MC_loss_3 = supervisor(x3,targets_3,height=7,cnum=136)  # c3=15, 15*136=2040
            MC_loss_4 = supervisor(x4,targets_4,height=7,cnum=20)  # c4=100, 20*100=2000

        x1, x2, x3, x4 = self.avg_p(x1), self.avg_p(x2), self.avg_p(x3), self.avg_p(x4)
        x1, x2, x3, x4 = x1.view(-1, seq, 2040), x2.view(-1, seq, 2000), x3.view(-1, seq, 2040), x4.view(-1, seq, 2000)  # bs//10,10,2040
        self.lstm1.flatten_parameters()  # 多GPU训练
        self.lstm2.flatten_parameters()
        y1, _ = self.lstm1(x1)  # bs//10, 10, 512
        y3, _ = self.lstm1(x3)
        y2, _ = self.lstm2(x2)
        y4, _ = self.lstm2(x4)

        y1, y2, y3, y4 = y1.contiguous().view(-1, 512), y2.contiguous().view(-1, 512), y3.contiguous().view(-1, 512), y4.contiguous().view(-1, 512)  # 10,512
        y1, y2, y3, y4 = F.relu(self.fc_h(y1)),F.relu(self.fc_h(y2)), F.relu(self.fc_h(y3)),F.relu(self.fc_h(y4))  # 10, 2040

        # x = self.max(x)
        # x = x.view(x.size(0), -1)
        y1, y2, y3, y4 = self.classifier_1(y1), self.classifier_2(y2), self.classifier_3(y3), self.classifier_4(y4)  # 10, 6
        
        targets_1, targets_2, targets_3, targets_4 = targets_1.data.float(), targets_2.data.float(), targets_3.data.float(), targets_4.data.float()
        loss_1, loss_2, loss_3 = criterion(y1, targets_1), criterion(y2, targets_2), criterion(y3, targets_3)
        loss_4 = criterion(y4, targets_4)
        total_loss = loss_1 + loss_2 + loss_3 + loss_4

        if self.training:
            return [y1, y2, y3, y4], total_loss, [MC_loss_1, MC_loss_2, MC_loss_3, MC_loss_4]
        else:
            return [y1, y2, y3, y4], total_loss


class model_lstm_cam(nn.Module):

    def __init__(self, feature_size=2048):

        super(model_lstm_cam, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.avg_p = resnet.avgpool
        self.lstm1 = nn.LSTM(2040, 512, batch_first=True)
        self.lstm2 = nn.LSTM(2000, 512, batch_first=True)
        self.fc_h = nn.Linear(512, 2040)
        # self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(512, 6), nn.Linear(512, 10), nn.Linear(512, 15), nn.Linear(512, 15)

        init.xavier_normal_(self.lstm1.all_weights[0][0])
        init.xavier_normal_(self.lstm1.all_weights[0][1])
        init.xavier_normal_(self.lstm2.all_weights[0][0])
        init.xavier_normal_(self.lstm2.all_weights[0][1])
        init.xavier_uniform_(self.fc_h.weight)

        self.features2fixed_1 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1,
                                          padding=0)
        self.features2fixed_2 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1,
                                          padding=0)
        self.features2fixed_3 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1,
                                          padding=0)
        self.features2fixed_4 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1,
                                          padding=0)

        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040 * 1 * 1
        self.classifier_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 6),
        )

        self.classifier_2 = nn.Sequential(
            nn.BatchNorm1d(2040 * 1 * 1),
            # nn.Dropout(0.5),
            nn.Linear(2040 * 1 * 1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 10),
        )

        self.classifier_3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 15),
        )

        self.classifier_4 = nn.Sequential(
            nn.BatchNorm1d(2040 * 1 * 1),
            # nn.Dropout(0.5),
            nn.Linear(2040 * 1 * 1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 100),
        )

    def forward(self, x):
        # pdb.set_trace()
        x = self.share(x)  # bs, 2048, 7, 7
        # x1 = self.features2fixed_1(x)  # bs, 2040, 7, 7
        # x2 = self.features2fixed_2(x)  # bs, 2000, 7, 7
        # x3 = self.features2fixed_3(x)  # bs, 2040, 7, 7
        x4 = self.features2fixed_4(x)  # bs, 2000, 7, 7
        seq = 10

        x4 = self.avg_p(x4)
        x4 = x4.view(-1,seq,2000)
        self.lstm2.flatten_parameters()
        y4, _ = self.lstm2(x4)
        y4 = y4.contiguous().view(-1,512)
        y4 = F.relu(self.fc_h(y4))
        y4 = self.classifier_4(y4)


        # x1, x2, x3, x4 = self.avg_p(x1), self.avg_p(x2), self.avg_p(x3), self.avg_p(x4)
        # x1, x2, x3, x4 = x1.view(-1, seq, 2040), x2.view(-1, seq, 2000), x3.view(-1, seq, 2040), x4.view(-1, seq,
        #                                                                                                  2000)  # bs//10,10,2040
        # self.lstm1.flatten_parameters()  # 多GPU训练
        # self.lstm2.flatten_parameters()
        # y1, _ = self.lstm1(x1)  # bs//10, 10, 512
        # y3, _ = self.lstm1(x3)
        # y2, _ = self.lstm2(x2)
        # y4, _ = self.lstm2(x4)
        #
        # y1, y2, y3, y4 = y1.contiguous().view(-1, 512), y2.contiguous().view(-1, 512), y3.contiguous().view(-1,
        #                                                                                                     512), y4.contiguous().view(
        #     -1, 512)  # 10,512
        # y1, y2, y3, y4 = F.relu(self.fc_h(y1)), F.relu(self.fc_h(y2)), F.relu(self.fc_h(y3)), F.relu(
        #     self.fc_h(y4))  # 10, 2040
        #
        # # x = self.max(x)
        # # x = x.view(x.size(0), -1)
        # y1, y2, y3, y4 = self.classifier_1(y1), self.classifier_2(y2), self.classifier_3(y3), self.classifier_4(
        #     y4)  # 10, 6

        # tmp = torch.sigmoid(y4)[0].detach().numpy()
        # tmp[tmp<0.5] = 0
        # tmp[tmp>0.5] = 1
        # print(np.nonzero(tmp))
        return y4


class model_all4(nn.Module):

    def __init__(self, feature_size=2048):

        super(model_all4, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed_1 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        self.features2fixed_2 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        self.features2fixed_3 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        self.features2fixed_4 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040*1*1
        self.classifier_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 6),
        )

        self.classifier_2 = nn.Sequential(
            nn.BatchNorm1d(2000*1*1),
            #nn.Dropout(0.5),
            nn.Linear(2000*1*1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 10),
        )

        self.classifier_3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 15),
        )

        self.classifier_4 = nn.Sequential(
            nn.BatchNorm1d(2000*1*1),
            #nn.Dropout(0.5),
            nn.Linear(2000*1*1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 100),
        )

    def forward(self, x, targets_1, targets_2, targets_3, targets_4, sequence_length):

        x = self.share(x)
        x1 = self.features2fixed_1(x)
        x2 = self.features2fixed_2(x)
        x3 = self.features2fixed_3(x)
        x4 = self.features2fixed_4(x)

        if self.training:
            MC_loss_1 = supervisor(x1,targets_1,height=7,cnum=340)
            MC_loss_2 = supervisor(x2,targets_2,height=7,cnum=200)
            MC_loss_3 = supervisor(x3,targets_3,height=7,cnum=136)
            MC_loss_4 = supervisor(x4,targets_4,height=7,cnum=20)

        x1, x2, x3, x4 = self.max(x1), self.max(x2), self.max(x3), self.max(x4)
        x1, x2, x3, x4 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1), x4.view(x4.size(0), -1)
        y1, y2, y3, y4 = self.classifier_1(x1), self.classifier_2(x2), self.classifier_3(x3), self.classifier_4(x4)
        
        targets_1, targets_2, targets_3, targets_4 = targets_1.data.float(), targets_2.data.float(), targets_3.data.float(), targets_4.data.float()
        loss_1, loss_2, loss_3 = criterion(y1, targets_1), criterion(y2, targets_2), criterion(y3, targets_3)
        loss_4 = criterion(y4, targets_4)
        total_loss = loss_1 + loss_2 + loss_3 + loss_4

        if self.training:
            return [y1, y2, y3, y4], total_loss, [MC_loss_1, MC_loss_2, MC_loss_3, MC_loss_4]
        else:
            return [y1, y2, y3, y4], total_loss


class model_all4_cam(nn.Module):

    def __init__(self, feature_size=2048):

        super(model_all4_cam, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed_1 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1,
                                          padding=0)
        self.features2fixed_2 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1,
                                          padding=0)
        self.features2fixed_3 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1,
                                          padding=0)
        self.features2fixed_4 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1,
                                          padding=0)

        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040 * 1 * 1
        self.classifier_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 6),
        )

        self.classifier_2 = nn.Sequential(
            nn.BatchNorm1d(2000 * 1 * 1),
            # nn.Dropout(0.5),
            nn.Linear(2000 * 1 * 1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 10),
        )

        self.classifier_3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            # nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 15),
        )

        self.classifier_4 = nn.Sequential(
            nn.BatchNorm1d(2000 * 1 * 1),
            # nn.Dropout(0.5),
            nn.Linear(2000 * 1 * 1, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(feature_size, 100),
        )

    def forward(self, x):

        x = self.share(x)
        x1 = self.features2fixed_1(x)
        x2 = self.features2fixed_2(x)
        x3 = self.features2fixed_3(x)
        x4 = self.features2fixed_4(x)

        x1, x2, x3, x4 = self.max(x1), self.max(x2), self.max(x3), self.max(x4)
        x1, x2, x3, x4 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1), x4.view(x4.size(0), -1)
        y1, y2, y3, y4 = self.classifier_1(x1), self.classifier_2(x2), self.classifier_3(x3), self.classifier_4(x4)

        # x4 = self.max(x4)
        # x4 = x4.view(x4.size(0), -1)
        # y4 = self.classifier_4(x4)
        return y4


class model_all(nn.Module):

    def __init__(self, feature_size=2048):

        super(model_all, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed_1 = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        self.features2fixed_2 = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040*1*1
        self.classifier_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 6),
        )

        self.classifier_2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 10),
        )

        self.classifier_3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, 15),
        )

    def forward(self, x, targets_1, targets_2, targets_3):

        x = self.share(x)
        x1 = self.features2fixed_1(x)
        x2 = self.features2fixed_2(x)
        x = self.features2fixed_1(x)

        if self.training:
            MC_loss_1 = supervisor(x1,targets_1,height=7,cnum=340)
            MC_loss_2 = supervisor(x2,targets_2,height=7,cnum=200)
            MC_loss_3 = supervisor(x1,targets_3,height=7,cnum=136)

        x = self.max(x)
        x = x.view(x.size(0), -1)
        y1, y2, y3 = self.classifier_1(x), self.classifier_2(x), self.classifier_3(x)
        
        targets_1, targets_2, targets_3 = targets_1.data.float(), targets_2.data.float(), targets_3.data.float()
        loss_1, loss_2, loss_3 = criterion(y1, targets_1), criterion(y2, targets_2), criterion(y3, targets_3)
        total_loss = loss_1 + loss_2 + loss_3

        if self.training:
            return [y1, y2, y3], total_loss, [MC_loss_1, MC_loss_2, MC_loss_3]
        else:
            return [y1, y2, y3], total_loss


class model_tool(nn.Module):

    def __init__(self, feature_size=2048,classes_num=6):

        super(model_tool, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040*1*1
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, targets):

        x = self.share(x)
        x = self.features2fixed(x)

        if self.training:
            cnums = [340, 200, 136, 20]
            MC_loss = supervisor(x,targets,height=7,cnum=cnums[0])

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        targets = targets.data.float()
        loss = criterion(x, targets)

        if self.training:
            return x, loss, MC_loss
        else:
            return x, loss


class model_verb(nn.Module):

    def __init__(self, feature_size=2048,classes_num=10):

        super(model_verb, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2000*1*1
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, targets):

        x = self.share(x)
        x = self.features2fixed(x)

        if self.training:
            cnums = [340, 200, 136, 20]
            MC_loss = supervisor(x,targets,height=7,cnum=cnums[1])

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        targets = targets.data.float()
        loss = criterion(x, targets)

        if self.training:
            return x, loss, MC_loss
        else:
            return x, loss


class model_target(nn.Module):

    def __init__(self, feature_size=2048,classes_num=15):

        super(model_target, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed = nn.Conv2d(in_channels=feature_size, out_channels=2040, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2040*1*1
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, targets):

        x = self.share(x)
        x = self.features2fixed(x)

        if self.training:
            cnums = [340, 200, 136, 20]
            MC_loss = supervisor(x,targets,height=7,cnum=cnums[2])

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        targets = targets.data.float()
        loss = criterion(x, targets)

        if self.training:
            return x, loss, MC_loss
        else:
            return x, loss


class model_triplet(nn.Module):

    def __init__(self, feature_size=2048,classes_num=100):

        super(model_triplet, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)

        self.features2fixed = nn.Conv2d(in_channels=feature_size, out_channels=2000, kernel_size=1, stride=1, padding=0)
        
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)

        self.num_ftrs = 2000*1*1
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            #nn.Dropout(0.5),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, targets):

        classes_num = targets.shape[1]

        x = self.share(x)
        x = self.features2fixed(x)

        if self.training:
            MC_loss = supervisor(x,targets,height=7,cnum=20)

        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        targets = targets.data.float()
        loss = criterion(x, targets)

        if self.training:
            return x, loss, MC_loss
        else:
            return x, loss


# if __name__ == '__main__':
    # classes=6 instrument
    # classes=10 verb
    # classes=15 target

    # data = torch.autograd.Variable(torch.rand(10, 3, 224, 224)).cuda()
    # targets_1, targets_2, targets_3 = torch.zeros(10, 6).long().cuda(), torch.zeros(10, 10).long().cuda(), torch.zeros(10, 15).long().cuda()
    # targets_4 = torch.zeros(10, 100).long().cuda()
    #
    # net = model_lstm()
    # net = net.cuda()
    # net.train()
    # out, loss, MC_loss = net(data, targets_1, targets_2, targets_3, targets_4)
    # # total_loss = loss + MC_loss[0] + MC_loss[1]
    # print(loss)