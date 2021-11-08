#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   baseline.py
@Contact :   rlihuichen@stu.scu.edu.cn
@License :   None

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
20-7-10 下午5:12   LihuiChen      1.0         None
'''

# import lib

import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inFe, outFe):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inFe, outFe, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outFe, outFe, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


class Net(nn.Module):
    def __init__(self, opt=None):
        super(Net, self).__init__()
        hid_dim = 64
        input_dim = 64
        num_resblock = 3
        self.num_cycle = 5

        self.wrapper = nn.Conv2d(1, hid_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(1, input_dim, 3, 1, 1)

        self.hidden_unit_forward_list = nn.ModuleList()
        self.hidden_unit_backward_list = nn.ModuleList()
        
        for _ in range(self.num_cycle):
            compress_1 = (nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_1 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_forward_list.append(nn.Sequential(compress_1, resblock_1))

            compress_2 = (nn.Conv2d(hid_dim + input_dim + hid_dim, hid_dim, 1, 1, 0))
            resblock_2 = nn.Sequential(*[
                ResBlock(hid_dim, hid_dim) for _ in range(num_resblock)
            ])
            self.hidden_unit_backward_list.append(nn.Sequential(compress_2, resblock_2))

        self.conv2 = nn.Conv2d(hid_dim, 1, 3, 1, 1)

    def forward(self, ms, pan, mask=None, is_cat_out=False):
        '''
        :param ms: LR ms images
        :param pan: pan images
        :param mask: mask to record the batch size of each band
        :return:
            HR_ms: a list of HR ms images,
        '''

        if mask is None:
            mask = [1 for _ in range(ms.shape[1])]
            is_cat_out = True

        ms = ms.split(1, dim=1)
        pan_state = self.wrapper(pan)
        hidden_state = pan_state
        blur_ms_list = []


        backward_hidden = []
        for idx, band in enumerate(ms):
            band = F.interpolate(band[:mask[idx]], scale_factor=4, mode='bicubic', align_corners=False)
            blur_ms_list.append(band)
            backward_hidden.append(self.conv1(band))

        backward_hidden = backward_hidden[::-1]
        for idx_cycle in range(self.num_cycle):
            ## forward recurrence
            forward_hidden = []
            for idx in range(len(blur_ms_list)):
                hidden_state = hidden_state[:mask[idx]]
                band = torch.cat((backward_hidden[-(idx+1)], hidden_state, pan_state[:mask[idx]]), dim=1)
                hidden_state = self.hidden_unit_forward_list[idx_cycle](band)
                forward_hidden.append(hidden_state)
            ## backward recurrence
            backward_hidden = []
            for idx in range(len(blur_ms_list)):
                start_pan_stat = hidden_state.shape[0]
                hidden_state = torch.cat((hidden_state, pan_state[start_pan_stat:mask[-(idx+1)]]),dim=0)
                band = torch.cat((forward_hidden[-(idx + 1)], hidden_state, pan_state[:mask[-(idx+1)]]), dim=1)
                hidden_state = self.hidden_unit_backward_list[idx_cycle](band)
                backward_hidden.append(hidden_state)

        HR_ms = []
        for idx in range(len(blur_ms_list)):
            band = self.conv2(backward_hidden[-(idx+1)])
            band = band + blur_ms_list[idx]
            HR_ms.append(band)
        return HR_ms if not is_cat_out else (HR_ms, torch.cat(HR_ms, dim=1))


class myloss(nn.Module):
    def __init__(self, opt=None):
        super(myloss, self).__init__()

    def forward(self, ms, HR, mask=None):
        diff = 0
        count = 0
        HR = torch.split(HR, 1, dim=1)
        if mask is None:
            mask = [1 for _ in range(len(HR))]
            ms = ms[0]
        for idx, (band, hr) in enumerate(zip(ms, HR)):
            b, t, h, w = band.shape
            count += b * t * h * w
            diff += torch.sum(torch.abs(band - hr[:mask[idx]]))
        return diff / count
