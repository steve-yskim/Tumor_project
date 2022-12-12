import os
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, \
    adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import math
from unet_model import *
import gc
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy import ndimage
import time

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow
import gc


# Reset Keras Session
def reset_keras(my_class_1, my_class_2, device_number):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    #     try:
    del my_class_1
    del my_class_2

    gc.collect()
    #     except:
    #         pass
    print(gc.collect())  # if it does something you should see a number as output
    # use the same config as you used to create the session


#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 1
#     config.gpu_options.visible_device_list = "0"
#     tensorflow.compat.v1.keras.backend.set_session(tensorflow.Session(config=config))

class grid_search:
    def __init__(self, file, model_list, batch_size_list, lr_list, focal_p_list, decay_list, max_epoch,
                 interval, train, initial_exp, start_exp, device_number):
        self.df_path = 'DATA/meta_train.csv'  # '../../2nd_particle_whole_zoomout_train.json' #
        self.df_val_path = 'DATA/meta_test.csv'  # '../../22nd_particle_whole_zoomout_val.json' #
        self.img_dir = 'DATA/train/x_512'
        self.img_raw_dir = 'DATA/train/x'
        self.mask_dir = 'DATA/train/y_512'
        self.x_col = 'img_fname'
        self.x_col_mask = 'mask_fname'
        self.data_gen_args = dict(rotation_range=45,
                                  shear_range=5,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  zoom_range=(1 / 1.2, 1.2),
                                  fill_mode='reflect',
                                  brightness_range=(0.95, 1.05), )
        self.data_gen_args_test = dict()

        self.file = file
        self.model_list = model_list
        self.batch_size_list = batch_size_list
        self.lr_list = lr_list
        self.focal_p_list = focal_p_list
        self.decay_list = decay_list
        self.max_epoch = max_epoch
        self.save_interval = interval
        self.train = train
        self.exp = initial_exp
        self.start_exp = start_exp
        self.run(device_number)

    def run(self, device_number):
        for md in self.model_list:
            for b in self.batch_size_list:
                for lr in self.lr_list:
                    for p in self.focal_p_list:
                        for d in self.decay_list:
                            model_type = md
                            batch_size = b
                            lr = lr
                            condition = '{}_md({})b({})L({})p({})d({})'.format(self.exp, md, b, lr, p, d)
                            file_path = os.path.join(self.file, condition)
                            if self.exp < self.start_exp:
                                print('{} passed'.format(condition))
                                self.exp = self.exp + 1
                                pass
                            else:
                                #                                 try :
                                print(condition)
                                self.my_unet = fine_tuning(self.exp, self.file, condition, md, p, lr, d,
                                                           self.save_interval, self.df_path,
                                                           self.df_val_path, self.img_dir, self.mask_dir, self.x_col,
                                                           self.x_col_mask,
                                                           self.data_gen_args, self.data_gen_args_test, b,
                                                           batch_size_test=1,
                                                           channel=1, max_epoch=self.max_epoch, early_stopping=None)

                                if self.train == True:
                                    self.my_unet.train()
                                    self.my_unet.save_history()
                                else:
                                    pass
                                self.my_unet.find_best_epoch()
                                self.my_unet_inf = inference(md, file_path, self.img_raw_dir, self.df_val_path,
                                                             self.my_unet.best_model_path,
                                                             channel=1, max_epoch=self.my_unet.max_epoch)
                                reset_keras(self.my_unet, self.my_unet_inf, device_number)
                                self.exp = self.exp + 1


#                                 except :
#                                     self.exp = self.exp + 1
#                                     pass


def saveResult(image_path, save_path, npyfile, flag_multi_class=False, num_class=1, as_gray=True,
               text="{}_predict.png"):
    Fig_list = os.listdir(image_path)
    for i, item in enumerate(npyfile):
        img_raw = io.imread(os.path.join(image_path, "{}".format(Fig_list[i])), as_gray=as_gray)
        img = cv2.resize(item, dsize=(img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_AREA)
        img = (255 * img).astype('uint8')
        #         img.trans.resize(img,(1280,1024))
        io.imsave(os.path.join(save_path, text.format(Fig_list[i].split('.')[0])), img)


class Load_dataset:
    def __init__(self, df_path, df_val_path, img_dir, mask_dir, x_col, x_col_mask,
                 data_gen_args, data_gen_args_test, batch_size, batch_size_test=16, channel=2):
        self.dataframe = pd.read_csv(df_path).iloc[:, 1:]
        self.dataframe_val = pd.read_csv(df_val_path).iloc[:, 1:]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.x_col = x_col
        self.x_col_mask = x_col_mask
        self.data_gen_args = data_gen_args
        self.data_gen_args_test = data_gen_args_test
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.channel = channel
        self.make_generator()
        self.size_train = len(self.dataframe)
        self.size_val = len(self.dataframe_val)

    def make_generator(self):
        self.myGene = self.trainGenerator_DF(self.batch_size, self.img_dir, self.mask_dir, self.dataframe, self.x_col,
                                             self.x_col_mask,
                                             self.data_gen_args, save_to_dir=None, shuffle=True, channel=self.channel)
        self.myGene_test = self.trainGenerator_DF(self.batch_size_test, self.img_dir, self.mask_dir, self.dataframe_val,
                                                  self.x_col, self.x_col_mask,
                                                  self.data_gen_args_test, save_to_dir=None, shuffle=False,
                                                  channel=self.channel)

    def trainGenerator_DF(self, batch_size, img_path, mask_path, dataframe, x_col, x_col_mask, aug_dict,
                          aug_dict_mask=None,
                          image_color_mode="rgb", mask_color_mode="grayscale", image_save_prefix="image",
                          mask_save_prefix="mask",
                          save_to_dir=None, target_size=(512, 512), seed=1, shuffle=True, channel=1):
        image_datagen = ImageDataGenerator(**aug_dict)
        image_generator = image_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=img_path,
            x_col=x_col,
            # y_col = y_col,
            # classes=[image_folder],
            class_mode=None,
            color_mode=image_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed,
            shuffle=shuffle)
        mask_generator = image_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=mask_path,
            x_col=x_col_mask,
            # y_col=y_col_mask,
            # classes=[mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix,
            seed=seed,
            shuffle=shuffle)
        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            img = img / 255
            mask = (mask / np.expand_dims((mask.max(axis=(1))).max(axis=1) + 0.0000000001, axis=(1, 2)))
            # mask = mask /255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            yield (img, mask[:, :, :, :channel])


class fine_tuning:
    def __init__(self, exp, file, condition, model, focal_p, lr, decay, save_interval, df_path, df_val_path, img_dir,
                 mask_dir, x_col, x_col_mask,
                 data_gen_args, data_gen_args_test, batch_size, batch_size_test=16, channel=1, max_epoch=2000,
                 early_stopping=None):
        self.exp = exp
        self.file = file
        self.condition = condition
        self.file_path = os.path.join(self.file, self.condition)
        self.epsilon = tf.keras.backend.epsilon()
        os.makedirs(self.file_path, exist_ok=True)
        self.max_epoch = max_epoch
        self.md = model
        self.p = focal_p
        self.b = batch_size
        self.d = decay
        self.lr = lr
        self.save_interval = save_interval
        self.channel = channel
        self.unet_data = Load_dataset(df_path, df_val_path, img_dir, mask_dir, x_col, x_col_mask,
                                      data_gen_args, data_gen_args_test, self.b, batch_size_test, self.channel)
        self.es = early_stopping  # EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500)

    #         self.train()

    def focal_loss(self, y_true, y_pred):
        return (tf.reduce_mean(-(self.p * y_true * tf.math.log(y_pred + self.epsilon)) - (
                    (1 - self.p) * (1 - y_true) * tf.math.log(1 - y_pred + self.epsilon))))

    def train(self):
        if self.md == 'unet':
            self.model = unet(decay=self.d, channel=self.channel)
        elif self.md == 'unet_mobile':
            self.model = unet_mobile(decay=self.d, channel=self.channel)
        self.model.compile(optimizer=Adam(lr=self.lr), loss=self.focal_loss,
                           metrics=['accuracy', p1, r1, f1, p2, r2, f2, mean_iou])
        checkpoint_path = self.file_path + '/{}_'.format(
            self.condition) + '{epoch:02d}-{val_loss:.3f}-{f1:.3f}-{val_f1:.3f}.h5'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                              save_weights_only=True, period=self.save_interval)
        callbacks = [model_checkpoint]
        if self.es:
            callbacks.append(self.es)
        self.history = self.model.fit(self.unet_data.myGene,
                                      steps_per_epoch=int(np.ceil(self.unet_data.size_train / self.b))
                                      , epochs=self.max_epoch, validation_data=self.unet_data.myGene_test,
                                      validation_steps=int(np.ceil(self.unet_data.size_val / self.b)),
                                      callbacks=callbacks)

    def save_history(self):
        self.show_history()
        self.save_summary()

    def show_history(self):
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])  # +history_2.history['loss'])
        plt.plot(self.history.history['val_loss'])  # +history_2.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(self.history.history['f1'])  # +history_2.history['f1'])
        plt.plot(self.history.history['val_f1'])  # +history_2.history['val_f1'] ,alpha = 0.5)
        plt.xlabel('Epoch')
        plt.ylabel('F1-socre')
        plt.savefig(os.path.join(self.file_path, 'Loss-f1_plot.png'))
        # plt.xlim(0,500)
        # plt.ylim(0.04,0.15)
        #         plt.show()
        self.history_df = pd.DataFrame(self.history.history)
        self.history_df.to_csv(os.path.join(self.file_path, 'history.csv'))

    def save_summary(self):
        A = np.array(self.history_df.index + 1)
        #         A_b = A[A<=600]
        #         A_o = A[A>600]
        #         index = list(A_b[A_b%100==0]-1)+list(A_o[A_o%300==0]-1)
        index = list(A[A % self.save_interval == 0] - 1)
        Result = self.history_df[['loss', 'val_loss', 'f1', 'val_f1', 'mean_iou', 'val_mean_iou']].iloc[index,
                 :].reset_index()
        Result['index'] = Result['index'] + 1
        Result.rename(columns={'index': 'epoch'}, inplace=True)
        Result['index'] = self.exp
        Result['batch_size'] = self.b
        Result['focal_p'] = self.p
        Result['learning_rate'] = self.lr
        Result['L2_decay'] = self.d
        Result = Result[
            ['index', 'batch_size', 'focal_p', 'learning_rate', 'L2_decay', 'epoch', 'loss', 'val_loss', 'f1', 'val_f1',
             'mean_iou', 'val_mean_iou']]
        if not ('Result.csv' in os.listdir(self.file)):
            Result.to_csv(os.path.join(self.file, 'Result.csv'))
        else:
            Result_0 = pd.read_csv(os.path.join(self.file, 'Result.csv'), index_col=0)
            Result = pd.concat([Result_0, Result], axis=0)
            Result.to_csv(os.path.join(self.file, 'Result.csv'))

    def find_best_epoch(self):
        targetPattern = r"{}/*.h5".format(self.file_path)
        h5_list = glob.glob(targetPattern)
        f1_list = []
        for h5 in h5_list:
            # f1_list.append(float(h5.split('-')[-2]))   # train 기준
            f1_list.append(float(h5.split('-')[-1][:5]))  # validation 기준
        self.best_model_path = h5_list[(np.argmax(np.array(f1_list)))]
        self.max_epoch = int((self.best_model_path.split('-')[-4].split('_'))[-1])


class inference:
    def __init__(self, model, file_path, img_root_path, df_test_path, model_path, channel=1, max_epoch=None):
        self.md = model
        self.dataframe_test = pd.read_csv(df_test_path).iloc[:, 1:]
        self.img_root_path = img_root_path
        self.file_path = file_path
        self.model_path = model_path
        self.channel = channel
        self.max_epoch = max_epoch
        self.predict()
        self.save_pred_img()

    def predict(self):
        if self.md == 'unet':
            self.model = unet(channel=self.channel)
        elif self.md == 'unet_mobile':
            self.model = unet_mobile(channel=self.channel)
        self.model.load_weights(self.model_path)
        self.img_arr = []
        self.img_arr_raw_list = []
        for img_fname in self.dataframe_test['img_fname']:
            img = cv2.imread(os.path.join(self.img_root_path, img_fname))  # , cv2.IMREAD_GRAYSCALE)
            self.img_arr_raw_list.append(img)
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
            self.img_arr.append(img)
        self.img_arr = np.array(self.img_arr) / 255
        self.results = self.model.predict(self.img_arr, verbose=1, batch_size=1)

    def save_pred_img(self):
        start = time.time()
        # print('watershed proceesing...')
        self.results[self.results < 0.5] = 0
        self.results[self.results >= 0.5] = 1
        for i, Result in (enumerate(self.results)):
            width = self.dataframe_test.iloc[i]['width']
            height = self.dataframe_test.iloc[i]['height']
            img_pred = (255 * Result[:, :, 0]).astype('uint8')
            img_pred = (cv2.resize(img_pred, dsize=(width, height), interpolation=cv2.INTER_AREA))  # [:cutrow,
            if self.max_epoch:
                pred_path = 'Inference@epoch={}'.format(self.max_epoch)
            else:
                pred_path = 'Inference'
            save_path = os.path.join(self.file_path, pred_path)
            save_name = Path(self.dataframe_test['img_fname'][i]).stem
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, save_name + '_Inference.jpg'), img_pred)
        print('Completed!!! {} sec'.format(time.time() - start))
