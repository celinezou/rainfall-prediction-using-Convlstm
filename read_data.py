#!/usr/bin/env python3

import time
import os
from PIL import Image
import numpy as np


class read_radar(object):

    def __init__(self, d_dir):
        self.raw_dir = d_dir
        self.saved_file_name = 'rawdata.npz'
        self.frames = 61
        self.train_data = None
        self.shifted_data = None
        self.row = 224
        self.col = 224
        self.interval = 6

    def read_file(self, file_name):
        print('Run task %s (%s)...' % (file_name, os.getpid()))
        start = time.time()
        file_data = []
        with open(file_name) as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                file_data.append(line.split(','))
                line = f.readline()
        end = time.time()
        print('Task %s runs %0.2f seconds.' % (file_name, (end - start)))
        return np.array(file_data, dtype=float)[40:70, 40:70]

    def generate_radarfsl(self):
        if os.path.isfile(self.saved_file_name):
            self.loadArray()
        else:
            k = 0
            for fp in os.listdir(self.raw_dir):
                data = []
                for i in range(self.frames):
                    fp2 = self.raw_dir + fp + '\\' + fp + '_%s.png' % str(i).zfill(3)
                    image = Image.open(fp2)
                    image = image.resize((224, 224))
                    data.append(np.array(image))
                data = np.array(data) * (1. / 255)
                train_data = np.zeros((31*200, self.interval, self.row, self.col, 1))
                shifted_data = np.zeros((31*200, self.interval, self.row, self.col, 1))
                for i in range(6):
                    train_data[0+31*k:31+31*k, i, :, :, 0] = data[5*i:5*i+31]
                    shifted_data[0+31*k:31+31*k, i, :, :, 0] = data[(i+1)*5:(i+1)*5+31]
                if k == 199:
                    break
                k += 1
            self.saveArray()

    def saveArray(self):
        np.savez(self.saved_file_name, X=self.train_data, Y=self.shifted_data)

    def loadArray(self):
        print('reading datas from saved file')
        arch = np.load(self.saved_file_name)
        self.train_data = arch['X']
        self.shifted_data = arch['Y']
        print('end reading data')
        print(np.shape(self.train_data))
        print(np.shape(self.shifted_data))


if __name__ == '__main__':
    data_dir = 'D:\\Radar_Competition\\SRAD2018_TRAIN_001\\'
    data_set = read_radar(data_dir)
    data_set.generate_radarfsl()
