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
import sys
sys.path.append('keras-unet-collection')
from keras_unet_collection import models as ku_model
from keras_unet_collection import losses as ku_loss
sys.path.append('keras-deeplab-v3-plus-master')
import model as dl_model
import gc
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy import ndimage
import time
import zipfile
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow
import gc
import shutil


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

def init_unet_model(model_name, input_shape, channel, backbone=None,  # 'EfficientNetB7',
                    freeze_backbone=False, freeze_batch_norm=False, decay=0):
    if model_name == 'unet':
        model = unet(decay=decay, channel=channel)

    elif model_name == 'unet_mobile':
        model = unet_mobile(decay=decay, channel=channel)

    elif model_name == 'unet_2d':
        model = ku_model.unet_2d(input_shape, [64, 128, 256, 512, 1024], n_labels=channel,
                                 stack_num_down=2, stack_num_up=1,
                                 activation='GELU', output_activation='Sigmoid',
                                 batch_norm=True, pool='max', unpool='nearest', name='unet2d')

    elif model_name == 'vnet':
        model = ku_model.vnet_2d(input_shape, filter_num=[16, 32, 64, 128, 256], n_labels=channel,
                                 res_num_ini=1, res_num_max=3,
                                 activation='PReLU', output_activation='Sigmoid',
                                 batch_norm=True, pool=False, unpool=False, name='vnet')


    elif model_name == 'r2unet':
        model = ku_model.r2_unet_2d(input_shape, [64, 128, 256, 512], n_labels=channel,
                                    stack_num_down=2, stack_num_up=1, recur_num=2,
                                    activation='ReLU', output_activation='Sigmoid',
                                    batch_norm=True, pool='max', unpool='nearest', name='r2unet')

    elif model_name == 'resunet':
        model = ku_model.resunet_a_2d(input_shape, [32, 64, 128, 256, 512, 1024],
                                      dilation_num=[1, 3, 15, 31],
                                      n_labels=channel, aspp_num_down=256, aspp_num_up=128,
                                      activation='ReLU', output_activation='Sigmoid',
                                      batch_norm=True, pool=False, unpool='nearest', name='resunet')

    # elif model_name == 'swin_unet':
    #     model = ku_model.swin_unet_2d(input_shape, filter_num_begin=64, n_labels=channel, depth=4,
    #                                   stack_num_down=2, stack_num_up=2, patch_size=(2, 2), num_heads=[4, 8, 8, 8],
    #                                   window_size=[4, 2, 2, 2], num_mlp=512, output_activation='Sigmoid',
    #                                   shift_window=True,
    #                                   name='swin_unet')

    elif model_name == 'swin_unet':
        model = ku_model.swin_unet_2d(input_shape, filter_num_begin=32, n_labels=channel, depth=2,# 4,
                                      stack_num_down=2, stack_num_up=2, patch_size=(2, 2), num_heads=[4, 8, 8, 8],
                                      window_size=[4, 2, 2, 2], num_mlp=512, output_activation='Sigmoid',
                                      shift_window=True,
                                      name='swin_unet')

    elif model_name == 'u2net':
        model = ku_model.u2net_2d(input_shape, n_labels=channel,
                                  filter_num_down=[64, 128, 256, 512], filter_num_up=[64, 64, 128, 256],
                                  filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128],
                                  filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256],
                                  activation='ReLU', output_activation=None,
                                  batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')

    # backbone 있는 모델

    elif model_name == 'unet_plus':
        model = ku_model.unet_plus_2d(input_shape, [64, 128, 256, 512], n_labels=channel,
                                      stack_num_down=2, stack_num_up=2,
                                      activation='LeakyReLU', output_activation='Sigmoid',
                                      batch_norm=False, pool='max', unpool=False, deep_supervision=True,
                                      backbone=backbone, weights='imagenet', freeze_backbone=freeze_backbone,
                                      freeze_batch_norm=freeze_batch_norm, name='xnet')

    elif model_name == 'unet_3plus':
        model = ku_model.unet_3plus_2d(input_shape, n_labels=channel, filter_num_down=[64, 128, 256, 512],
                                       filter_num_skip='auto', filter_num_aggregate='auto',
                                       stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                                       batch_norm=True, pool='max', unpool=False, deep_supervision=True,
                                       backbone=backbone, weights='imagenet', freeze_backbone=freeze_backbone,
                                       freeze_batch_norm=freeze_batch_norm,
                                       name='unet3plus')

    elif model_name == 'transunet':
        model = ku_model.transunet_2d(input_shape, filter_num=[64, 128, 256, 512], n_labels=channel, stack_num_down=2,
                                      stack_num_up=2, embed_dim=768, num_mlp=3072, num_heads=3, num_transformer=3,
                                      # 원래 12개
                                      activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid',
                                      batch_norm=True, pool=True, unpool='bilinear',
                                      backbone=backbone, weights='imagenet', freeze_backbone=freeze_backbone,
                                      freeze_batch_norm=freeze_batch_norm, name='transunet')

    elif model_name == 'att_unet':
        model = ku_model.att_unet_2d(input_shape, [64, 128, 256, 512], n_labels=channel,
                                     stack_num_down=2, stack_num_up=2,
                                     activation='ReLU', atten_activation='ReLU', attention='add',
                                     output_activation='Sigmoid',
                                     batch_norm=True, pool=False, unpool='bilinear',
                                     backbone=backbone, weights='imagenet', freeze_backbone=freeze_backbone,
                                     freeze_batch_norm=freeze_batch_norm, name='attunet')
    elif model_name == 'dl3_mobilenetv2':
        model = dl_model.Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=input_shape, classes=channel,
                                   backbone='mobilenetv2', OS=16, alpha=1., activation='sigmoid')

    elif model_name == 'dl3_xception':
        model = dl_model.Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=input_shape, classes=channel,
                                   backbone='xception', OS=16, alpha=1., activation='sigmoid')


    return model


def export_pred_img(dataframe_test, results, file_path, zip_path, cv_k=None, pred_path='submission', thr=0.5):
    start = time.time()
    results[results < thr] = 0
    results[results >= thr] = 1
    for i, Result in (enumerate(results)):
        width = dataframe_test.iloc[i]['width']
        height = dataframe_test.iloc[i]['height']
        img_pred = (255 * Result[:, :, 0]).astype('uint8')
        # img_pred =  Result[:, :, 0]
        img_pred = (cv2.resize(img_pred, dsize=(width, height), interpolation=cv2.INTER_AREA))  # [:cutrow,
        save_path = os.path.join(file_path, pred_path)
        save_name = Path(dataframe_test['img_fname'][i]).stem
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, save_name + '.png'), img_pred)
    print('Completed!!! {} sec'.format(time.time() - start))

    if cv_k is not None:
        zip_file = zipfile.ZipFile(os.path.join(zip_path, "submission_{}.zip".format(cv_k)), "w")  # "w": write 모드
    else:
        zip_file = zipfile.ZipFile(os.path.join(zip_path, "submission_enss.zip"), "w")  # "w": write 모드
    for file in os.listdir(save_path):
        if file.endswith('.png'):
            shutil.copyfile(os.path.join(save_path, file), file)
            zip_file.write(file, compress_type=zipfile.ZIP_DEFLATED)
            os.remove(file)
    zip_file.close()


class grid_search:
    def __init__(self, df_path, img_root, mask_root, img_test_root, file, cv_k_list, model_list, batch_size_list,
                 lr_list, focal_p_list, decay_list, max_epoch, interval, train, initial_exp, start_exp, device_number):
        self.df_path = df_path
        self.img_dir = img_root
        self.img_raw_dir = img_test_root
        self.mask_dir = mask_root
        #         self.df_path = 'meta_train.csv'
        #         self.df_val_path ='meta_test.csv'
        #         self.img_dir = 'DATA/train/x_512'
        #         self.img_raw_dir = 'DATA/train/x'
        #         self.mask_dir = 'DATA/train/y_512'
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
        self.cv_k_list = cv_k_list
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

                            self.Result = dict()
                            self.Result_list = []
                            self.Result['index'] = self.exp
                            self.Result['model'] = md
                            self.Result['batch_size'] = b
                            self.Result['learning_rate'] = lr
                            self.Result['L2_decay'] = d
                            self.Result['focal_p'] = p
                            condition = '{}_md({})b({})L({})p({})d({})'.format(self.exp, md, b, lr, p, d)
                            file_path = os.path.join(self.file, condition)
                            if self.exp < self.start_exp:
                                print('{} passed'.format(condition))
                                self.exp = self.exp + 1
                                pass
                            else:
                                # try:
                                for cv_k in self.cv_k_list:
                                    #                                 try :
                                    print(condition)
                                    self.my_unet = fine_tuning(self.exp, self.file, condition, cv_k, md, p, lr, d,
                                                               self.save_interval,
                                                               self.df_path, self.img_dir, self.mask_dir,
                                                               self.x_col,
                                                               self.x_col_mask, self.data_gen_args,
                                                               self.data_gen_args_test, b,
                                                               batch_size_test=1, channel=1,
                                                               max_epoch=self.max_epoch,
                                                               early_stopping=None)

                                    if self.train == True:
                                        if 'history.csv' in os.listdir(self.my_unet.file_k_path):
                                            print('Already trained, pass!!!')
                                        else:
                                            self.my_unet.train()
                                            self.my_unet.save_history()
                                    else:
                                        pass
                                    max_f1 = self.my_unet.find_best_epoch()
                                    self.Result['f1_cv_{}'.format(cv_k)] = max_f1
                                    self.my_unet_inf = inference(md, file_path, cv_k, self.img_raw_dir,
                                                                 self.my_unet.best_model_path, channel=1,
                                                                 max_epoch=self.my_unet.max_epoch)

                                    if cv_k == 0:
                                        results_enss = self.my_unet_inf.results.copy()
                                        k = 1
                                    else:
                                        results_enss = results_enss + self.my_unet_inf.results.copy()
                                        k += 1

                                results_enss = results_enss / k
                                export_pred_img(self.my_unet_inf.dataframe_test, results_enss, file_path, file_path,
                                                cv_k=None,
                                                pred_path='submission_enss', thr=0.5)

                                reset_keras(self.my_unet, self.my_unet_inf, device_number)
                                self.exp = self.exp + 1

                                # except:
                                #     sess = get_session()
                                #     clear_session()
                                #     sess.close()
                                #     sess = get_session()
                                #
                                #     try:
                                #         del my_class_1
                                #         del my_class_2
                                #     except:
                                #         pass
                                #
                                #     self.exp = self.exp + 1
                                #
                                #     pass


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
    def __init__(self, df_path, cv_k, img_dir, mask_dir, x_col, x_col_mask,
                 data_gen_args, data_gen_args_test, batch_size, batch_size_test=16, channel=2):
        dataframe_raw = pd.read_csv(df_path).iloc[:, 1:]
        # self.dataframe_val = pd.read_csv(df_val_path).iloc[:,1:]
        self.dataframe = dataframe_raw[dataframe_raw['kfold'] != cv_k]
        self.dataframe_val = dataframe_raw[dataframe_raw['kfold'] == cv_k]

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
    def __init__(self, exp, file, condition, cv_k, model, focal_p, lr, decay, save_interval, df_path, img_dir, mask_dir,
                 x_col, x_col_mask, data_gen_args, data_gen_args_test, batch_size, batch_size_test=16, channel=1,
                 max_epoch=2000,
                 early_stopping=None):
        self.exp = exp
        self.file = file
        self.condition = condition
        self.cv_k = cv_k
        self.file_path = os.path.join(self.file, self.condition)
        self.file_k_path = os.path.join(self.file_path, str(self.cv_k))
        self.epsilon = tf.keras.backend.epsilon()
        os.makedirs(self.file_k_path, exist_ok=True)
        self.max_epoch = max_epoch
        self.md = model
        self.p = focal_p
        self.b = batch_size
        self.d = decay
        self.lr = lr
        self.save_interval = save_interval
        self.channel = channel
        self.unet_data = Load_dataset(df_path, self.cv_k, img_dir, mask_dir, x_col, x_col_mask,
                                      data_gen_args, data_gen_args_test, self.b, batch_size_test, self.channel)
        self.es = early_stopping  # EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500)
        self.input_shape = (512, 512, 3)

    #         self.train()

    def focal_loss(self, y_true, y_pred):
        return (tf.reduce_mean(-(self.p * y_true * tf.math.log(y_pred + self.epsilon)) - (
                    (1 - self.p) * (1 - y_true) * tf.math.log(1 - y_pred + self.epsilon))))

    def train(self):
        self.model = init_unet_model(self.md, self.input_shape, self.channel, decay=self.d)
        self.model.compile(optimizer=Adam(lr=self.lr), loss=ku_loss.iou_seg,  # self.focal_loss,
                           metrics=['accuracy', f1, mean_iou])
        checkpoint_path = self.file_k_path + '/{}_'.format(
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
        # use_multiprocessing=True, workers=2)

    def save_history(self):
        self.show_history()
        self.save_summary()

    def show_history(self):
        fig = plt.figure(figsize=(19, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.plot(self.history.history['loss'])  # +history_2.history['loss'])
        plt.plot(self.history.history['val_loss'])  # +history_2.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        ax2 = fig.add_subplot(1, 3, 2)
        plt.plot(self.history.history['f1'])  # +history_2.history['f1'])
        plt.plot(self.history.history['val_f1'])  # +history_2.history['val_f1'] ,alpha = 0.5)
        plt.xlabel('Epoch')
        plt.ylabel('F1-socre')
        ax3 = fig.add_subplot(1, 3, 3)
        plt.plot(self.history.history['mean_iou'])  # +history_2.history['f1'])
        plt.plot(self.history.history['val_mean_iou'])  # +history_2.history['val_f1'] ,alpha = 0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Mean_iou')

        plt.savefig(os.path.join(self.file_k_path, 'Loss-f1-iou_plot.png'))
        # plt.xlim(0,500)
        # plt.ylim(0.04,0.15)
        #         plt.show()
        self.history_df = pd.DataFrame(self.history.history)
        self.history_df.to_csv(os.path.join(self.file_k_path, 'history.csv'))

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
        targetPattern = r"{}/*.h5".format(self.file_k_path)
        self.h5_list = glob.glob(targetPattern)
        f1_list = []
        for h5 in self.h5_list:
            # f1_list.append(float(h5.split('-')[-2]))   # train 기준
            f1_list.append(float(h5.split('-')[-1][:5]))  # validation 기준
        self.best_model_path = self.h5_list[(np.argmax(np.array(f1_list)))]
        self.max_epoch = int((self.best_model_path.split('-')[-4].split('_'))[-1])
        self.remove_rest_h5()
        return np.max(np.array(f1_list))

    def remove_rest_h5(self):
        for h5 in self.h5_list:
            if h5 == self.best_model_path:
                pass
            else:
                print(os.path.basename(h5) + ' removed')
                os.remove(h5)


class inference:
    def __init__(self, model, file_path, cv_k, img_root_path, model_path, channel=1, max_epoch=None):
        self.md = model
        # self.dataframe_test = pd.read_csv(df_test_path).iloc[:,1:]

        self.test_img_list = sorted(glob.glob(os.path.join(img_root_path, '*.jpg')))  # change to jpg
        # print(self.test_img_list)

        self.img_root_path = img_root_path
        self.file_path = file_path
        self.file_k_path = os.path.join(self.file_path, str(cv_k))
        self.cv_k = cv_k
        self.model_path = model_path
        self.channel = channel
        self.max_epoch = max_epoch
        self.input_shape = (512, 512, 3)
        self.predict()
        self.save_pred_img()
        #         self.export_pred_img(self.results, 0.5)
        export_pred_img(self.dataframe_test, self.results, self.file_k_path, self.file_path, cv_k=self.cv_k,
                        pred_path='submission_{}'.format(self.cv_k), thr=0.5)

    def predict(self):
        #         if self.md == 'unet' :
        #             self.model = unet(channel = self.channel)
        #         elif self.md == 'unet_mobile' :
        #             self.model = unet_mobile(channel = self.channel)
        self.model = init_unet_model(self.md, self.input_shape, self.channel, decay=0)
        self.model.load_weights(self.model_path)
        self.img_arr = []
        self.img_arr_raw_list = []
        # for img_fname in self.dataframe_test['img_fname'] :

        test_dict_list = []
        for img_fname in self.test_img_list:
            test_dict = {}
            img = cv2.imread(img_fname)  # , cv2.IMREAD_GRAYSCALE)
            test_dict['img_fname'] = os.path.basename(img_fname)
            test_dict['width'] = img.shape[1]
            test_dict['height'] = img.shape[0]
            test_dict_list.append(test_dict)
            self.img_arr_raw_list.append(img)
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
            self.img_arr.append(img)
        self.dataframe_test = pd.DataFrame(test_dict_list)
        self.img_arr = np.array(self.img_arr) / 255
        self.results = self.model.predict(self.img_arr, verbose=1, batch_size=1)

    def save_pred_img(self):
        start = time.time()
        # self.results[self.results < 0.5] = 0
        # self.results[self.results >= 0.5] = 1
        for i, Result in (enumerate(self.results)):
            width = self.dataframe_test.iloc[i]['width']
            height = self.dataframe_test.iloc[i]['height']
            img_pred = (255 * Result[:, :, 0]).astype('uint8')
            # img_pred =  Result[:, :, 0]
            img_pred = (cv2.resize(img_pred, dsize=(width, height), interpolation=cv2.INTER_AREA))  # [:cutrow,
            if self.max_epoch:
                pred_path = 'Inference@epoch={}'.format(self.max_epoch)
            else:
                pred_path = 'Inference'
            save_path = os.path.join(self.file_k_path, pred_path)
            save_name = Path(self.dataframe_test['img_fname'][i]).stem
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, save_name + '_Inference.jpg'), img_pred)
        print('Completed!!! {} sec'.format(time.time() - start))

    def export_pred_img(self, results, thr=0.5):
        start = time.time()
        results[results < thr] = 0
        results[results >= thr] = 1
        for i, Result in (enumerate(results)):
            width = self.dataframe_test.iloc[i]['width']
            height = self.dataframe_test.iloc[i]['height']
            img_pred = (255 * Result[:, :, 0]).astype('uint8')
            # img_pred =  Result[:, :, 0]
            img_pred = (cv2.resize(img_pred, dsize=(width, height), interpolation=cv2.INTER_AREA))  # [:cutrow,
            pred_path = 'submission'
            save_path = os.path.join(self.file_k_path, pred_path)
            save_name = Path(self.dataframe_test['img_fname'][i]).stem
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, save_name + '.png'), img_pred)
        print('Completed!!! {} sec'.format(time.time() - start))

        zip_file = zipfile.ZipFile(os.path.join(self.file_path, "submission_{}.zip".format(self.cv_k)),
                                   "w")  # "w": write 모드
        for file in os.listdir(save_path):
            if file.endswith('.png'):
                shutil.copyfile(os.path.join(save_path, file), file)
                zip_file.write(file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(file)
        zip_file.close()

#         # Compress
#         zip_file = zipfile.ZipFile(os.path.join(self.file_path, 'submission.zip'), 'w')
#         for file in os.listdir(save_path) :
#             if file.endswith('.png'):
#                 zip_file.write(os.path.join(file, compress_type=zipfile.ZIP_DEFLATED)
#         zip_file.close()


