#!/usr/bin/env python3

import time
import os
from PIL import Image
import numpy as np


class read_radar(object):

    def __init__(self, d_dir, f_dir):
        self.raw_dir = d_dir
        self.f_dir = f_dir
        self.frames = 61
        self.train_data = None
        self.shifted_data = None
        self.row = 50
        self.col = 50
        self.interval = 6

    def generate_radarfsl(self):
        data = []
        for i in range(self.frames):
            fp2 = self.raw_dir + self.f_dir + '\\' + self.f_dir + '_%s.png' % str(i).zfill(3)
            image = Image.open(fp2)
            image = image.resize((50, 50))
            data.append(np.array(image))
        data = np.array(data) * (1. / 255)
        train_data = np.zeros((31, self.interval, self.row, self.col, 1))
        shifted_data = np.zeros((31, self.interval, self.row, self.col, 1))
        for i in range(6):
            train_data[:, i, :, :, 0] = data[5*i:5*i+31]
            shifted_data[:, i, :, :, 0] = data[(i+1)*5:(i+1)*5+31]
#        return train_data[0:31:2], shifted_data[0:31:2]
        return train_data[0:31:5], shifted_data[0:31:5]
