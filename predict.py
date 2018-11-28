from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = load_model('D:\\Radar_Competition\\lstm\\LSTM.008.h5')
data_path_2 = 'D:\\Radar_Competition\\SRAD2018_TRAIN_002\\'
filepath_val = [fp for fp in os.listdir(data_path_2)]
input_frames = 6
s_select = 0
output_frames = 6
input_data = []
for i in range(0, 31, 5):
    fp2 = data_path_2 + filepath_val[s_select] + '\\' + filepath_val[s_select] + '_%s.png' % str(i).zfill(3)
    image = Image.open(fp2)
    image = image.resize((50, 50))
    input_data.append(np.array(image))
input_data = np.array(input_data) * (1. / 255)
input_data = input_data[1:, :, :, np.newaxis]

x_save = []
for i in range(output_frames):
    output = model.predict(input_data[np.newaxis, :, :, :, :])
    x_save.append(output[0, -1, :, :, :])
    input_data = np.concatenate((input_data, output[0, -1, :, :, :][np.newaxis, :, :, :]), axis=0)
    input_data = input_data[1:]
x_save = np.array(x_save)

for i in range(output_frames):
    image = x_save[i]*255.
    image = Image.fromarray(image[:, :, 0].astype(np.uint8))
    #image = image.resize((501, 501))
    image.show()
    image.save('.\\lstm\\predp_%s.png' % str(35+i*5))
