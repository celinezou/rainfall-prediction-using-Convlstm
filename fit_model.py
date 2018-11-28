import os
from matplotlib.pyplot import savefig, plot
import matplotlib.pyplot as plt
import numpy as np
from read_radar import read_radar
from random import randint
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPool3D
from keras.layers import LSTM

data_path_1 = 'D:\\Radar_Competition\\SRAD2018_TRAIN_001\\'
data_path_2 = 'D:\\Radar_Competition\\SRAD2018_TRAIN_002\\'
filepath = [fp for fp in os.listdir(data_path_1)]
filepath_val = [fp for fp in os.listdir(data_path_2)]
batch_size = 1
L = len(filepath)


def batch_generator(d_p, filepath, batch_size):
    while 1:
        train_datas = np.zeros((7 * batch_size, 6, 50, 50, 1))
        shifted_datas = np.zeros((7 * batch_size, 6, 50, 50, 1))
        for b in range(batch_size):
            i = randint(0, L-1)
            d = read_radar(d_p, filepath[i])
            train_data, shifted_data = d.generate_radarfsl()
            train_datas[0+7*b:7+7*b, :, :, :, :] = train_data
            shifted_datas[0 + 7 * b:7 + 7 * b, :, :, :, :] = shifted_data
        yield train_datas, shifted_datas


# --------build model-------------------------------------------
def Convlstm():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), input_shape=(None, 250, 250, 1), padding='same',
                       return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=4, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

    seq.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    seq.summary()
    return seq

#  input_shape's first dim should be "None", which means you can use any steps to predict.
def lstm():
    seq = Sequential()
    seq.add(Conv3D(filters=16, kernel_size=(3, 3, 3), input_shape=(None, 50, 50, 1), padding='same', activation='sigmoid'))
    seq.add(BatchNormalization())

    seq.add(MaxPool3D(pool_size=(1, 2, 2)))
    seq.add(Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='sigmoid'))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid'))
    seq.add(BatchNormalization())

    seq.add(Reshape((6, 25*25)))
    seq.add(LSTM(units=25*25, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    seq.add(LSTM(units=50 * 50, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))

    seq.add(Reshape((6, 50, 50, 1)))
    seq.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    seq.summary()
    return seq


checkpoint = ModelCheckpoint(filepath='D:\\Radar_Competition\\lstm\\LSTM.{epoch:03d}.h5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)


callbacks = [checkpoint, lr_reducer]
model = lstm()
history = model.fit_generator(generator=batch_generator(data_path_1, filepath, batch_size), steps_per_epoch=L/batch_size,
                                epochs=10, verbose=1, validation_data=batch_generator(data_path_2, filepath_val, batch_size),
                                validation_steps=1000, workers=5, callbacks=callbacks)

model.save('D:\\Radar_Competition\\lstm\\final.h5')
# summarize history for accuracy
plot(history.history['acc'])
plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
savefig('D:\\Radar_Competition\\lstm\\model_acc.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
savefig('D:\\Radar_Competition\\lstm\\model_loss.png')

