import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras import backend as keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import *

epsilon = tf.keras.backend.epsilon()


def limit_gpu(lim=True):
    if lim == True:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        pass


def focal_loss(y_true, y_pred):
    p = 0.99
    return (tf.reduce_mean(-(p * y_true * tf.math.log(y_pred)) - ((1 - p) * (1 - y_true) * tf.math.log(1 - y_pred))))


def focal_loss_2(y_true, y_pred):
    p1 = 0.85
    p2 = 0.5
    #     return (0.5*tf.reduce_mean(-(p1*y_true[:,0,:,:]*tf.math.log(y_pred[:,0,:,:]))-((1-p1)*(1-y_true[:,0,:,:])*tf.math.log(1-y_pred[:,0,:,:]))-(p2*y_true[:,1,:,:]*tf.math.log(y_pred[:,1,:,:]))-((1-p2)*(1-y_true[:,1,:,:])*tf.math.log(1-y_pred[:,1,:,:]))))
    return (tf.reduce_mean(-(p1 * y_true[:, :, :, 0] * tf.math.log(y_pred[:, :, :, 0])) - (
                (1 - p1) * (1 - y_true[:, :, :, 0]) * tf.math.log(1 - y_pred[:, :, :, 0])) - (
                                       1 * y_true[:, :, :, 1] * tf.math.log(y_pred[:, :, :, 1])) - (
                           (1 * (1 - y_true[:, :, :, 1]) * tf.math.log(1 - y_pred[:, :, :, 1])))))


import tensorflow.keras.backend as K


def p1(y_true, y_pred):
    #    y_pred = tf.dtypes.cast(tf.where(tf.less_equal(y_pred, tf.constant(0.5)), 0, 1), tf.float32)
    #     recall = (tf.reduce_sum(y_true*y_pred))/(tf.reduce_sum(y_true))
    y_true = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    precision = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_pred) + epsilon)
    #     f1 = 2*precision*recall/(precision+recall)
    return precision


def r1(y_true, y_pred):
    y_true = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    # y_pred = tf.dtypes.cast(tf.where(tf.less_equal(y_pred, tf.constant(0.5)), 0, 1), tf.float32)
    recall = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + epsilon)
    #     precision = (tf.reduce_sum(y_true*y_pred))/(tf.reduce_sum(y_pred))
    #     f1 = 2*precision*recall/(precision+recall)
    return recall


def f1(y_true, y_pred):
    y_true = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    #     y_pred = tf.dtypes.cast(tf.where(tf.less_equal(y_pred, tf.constant(0.5)), 0, 1), tf.float32)
    recall = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + epsilon)
    precision = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_pred) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return f1


def p2(y_true, y_pred):
    y_true = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    # y_pred = tf.dtypes.cast(tf.where(tf.less_equal(y_pred, tf.constant(0.5)), 0, 1), tf.float32)
    y_true = -y_true + 1
    y_pred = -y_pred + 1
    #     recall = (tf.reduce_sum(y_true*y_pred))/(tf.reduce_sum(y_true))
    precision = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_pred) + epsilon)
    #     f1 = 2*precision*recall/(precision+recall)
    return precision


def r2(y_true, y_pred):
    y_true = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    # y_pred = tf.dtypes.cast(tf.where(tf.less_equal(y_pred, tf.constant(0.5)), 0, 1), tf.float32)
    y_true = -y_true + 1
    y_pred = -y_pred + 1
    recall = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + epsilon)
    #     precision = (tf.reduce_sum(y_true*y_pred))/(tf.reduce_sum(y_pred))
    #     f1 = 2*precision*recall/(precision+recall)
    return recall


def f2(y_true, y_pred):
    y_true = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    # y_pred = tf.dtypes.cast(tf.where(tf.less_equal(y_pred, tf.constant(0.5)), 0, 1), tf.float32)
    y_true = -y_true + 1
    y_pred = -y_pred + 1
    recall = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + epsilon)
    precision = (tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_pred) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return f1


def recall_0(y_true, y_pred):
    y_true_0 = -y_true + 1
    y_pred_0 = -y_pred + 1
    return (tf.reduce_sum(y_true_0 * y_pred_0)) / (tf.reduce_sum(y_true_0) + epsilon)


def recall_loss(y_true, y_pred):
    return -0.5 * tf.math.log(recall_1(y_true, y_pred) + epsilon) - 0.5 * tf.math.log(
        recall_0(y_true, y_pred) + epsilon)


def mean_iou(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')  # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))


def unet(pretrained_weights=None, input_size=(512, 512, 3), decay=0, channel=1):
    inputs = Input(input_size)
    batch1 = BatchNormalization()(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(batch1)
    conv1 = Conv2D(64, 3, padding='same', kernel_regularizer=l2(decay))(conv1)
    batch1 = BatchNormalization()(conv1)
    batch1 = Activation('relu')(batch1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)  # (128,128,64)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(pool1)
    conv2 = Conv2D(128, 3, padding='same', kernel_regularizer=l2(decay))(conv2)
    batch2 = BatchNormalization()(conv2)
    batch2 = Activation('relu')(batch2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)  # (64,64,128)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(pool2)
    conv3 = Conv2D(256, 3, padding='same', kernel_regularizer=l2(decay))(conv3)
    batch3 = BatchNormalization()(conv3)
    batch3 = Activation('relu')(batch3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)  # (32,32,256)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(pool3)
    conv4 = Conv2D(512, 3, padding='same', kernel_regularizer=l2(decay))(conv4)
    batch4 = BatchNormalization()(conv4)
    batch4 = Activation('relu')(batch4)
    drop4 = Dropout(0.5)(batch4)  # (32, 32, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # (16,16,512)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(pool4)
    conv5 = Conv2D(1024, 3, padding='same', kernel_regularizer=l2(decay))(conv5)
    batch5 = BatchNormalization()(conv5)
    batch5 = Activation('relu')(batch5)
    drop5 = Dropout(0.5)(batch5)  # (16,16,1024)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_regularizer=l2(decay))(
        UpSampling2D(size=(2, 2))(conv5))  # (32, 32, 512)
    merge6 = concatenate([conv4, up6], axis=3)  # (32, 32, 768)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(merge6)
    conv6 = Conv2D(512, 3, padding='same', kernel_regularizer=l2(decay))(conv6)  # (32, 32, 512)
    batch6 = BatchNormalization()(conv6)
    batch6 = Activation('relu')(batch6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_regularizer=l2(decay))(
        UpSampling2D(size=(2, 2))(batch6))  # (64, 64, 256)
    merge7 = concatenate([conv3, up7], axis=3)  # (64, 64, 384)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(merge7)
    conv7 = Conv2D(256, 3, padding='same', kernel_regularizer=l2(decay))(conv7)  # (64, 64, 256)
    batch7 = BatchNormalization()(conv7)
    batch7 = Activation('relu')(batch7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_regularizer=l2(decay))(
        UpSampling2D(size=(2, 2))(batch7))  # (128, 128, 128)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(merge8)
    conv8 = Conv2D(128, 3, padding='same', kernel_regularizer=l2(decay))(conv8)
    batch8 = BatchNormalization()(conv8)
    batch8 = Activation('relu')(batch8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(decay))(
        UpSampling2D(size=(2, 2))(batch8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(merge9)
    conv9 = Conv2D(64, 3, padding='same', kernel_regularizer=l2(decay))(conv9)
    batch9 = BatchNormalization()(conv9)
    batch9 = Activation('relu')(batch9)
    conv10 = Conv2D(channel, 3, activation='relu', padding='same', kernel_regularizer=l2(decay))(batch9)
    conv10 = Conv2D(channel, 1, activation='sigmoid', kernel_regularizer=l2(decay))(conv10)
    model = Model(inputs=inputs, outputs=conv10)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model


# Mobile U-net
def encoding(pretrained_net='MobileNet'):
    #     input_Layer= tf.keras.layers.Input(shape=(224, 224,3))
    # premodel = eval("{}(weights = None, include_top=False, input_shape=(512,512,1))".format(pretrained_net))
    # premodel_clr = eval("{}(weights = 'imagenet', include_top=False, input_shape=(512, 512,3))".format(pretrained_net))
    # model_clr_weight = premodel_clr.get_weights()
    # model_gray_weight = [np.expand_dims(np.average(model_clr_weight[0], axis=2), axis=2)] + model_clr_weight[1:]
    # premodel.set_weights(model_gray_weight)
    premodel = tf.keras.models.load_model('Premodel.h5')

    #     premodel.load_weigts('../experiment_50x_set2_aug_transfer(128)/11_m(MobileNet)b(128)L(1e-05)/11_m(MobileNet)b(128)L(1e-05)_200-0.080-0.060-0.975-1.000.h5')
    def cut_pretrained_model(shape, i, f):
        input_Layer = tf.keras.layers.Input(shape=shape)
        x = input_Layer
        for layer in premodel.layers[i:f]:
            x = layer(x)
        Out_Layer = x
        model = tf.keras.Model(inputs=[input_Layer], outputs=[Out_Layer])
        return model

    model_1 = cut_pretrained_model((128 * 4, 128 * 4, 1), 0, 11)
    model_2 = cut_pretrained_model((64 * 4, 64 * 4, 64), 11, 24)
    model_3 = cut_pretrained_model((32 * 4, 32 * 4, 128), 24, 37)
    model_4 = cut_pretrained_model((16 * 4, 16 * 4, 256), 37, 74)
    model_5 = cut_pretrained_model((8 * 4, 8 * 4, 512), 74, -1)

    return model_1, model_2, model_3, model_4, model_5


def decoding_layer(x, C1, C2, C3, C4, decay=0):
    add_ratio = 0.5
    x = tf.keras.layers.Conv2D(512, (1, 1))(x)  # (4, 4, 512)
    x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
                               add_ratio * tf.keras.layers.Conv2D(512, (1, 1), kernel_regularizer=l2(decay))(C4)])
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)  # (8, 8, 512)
    x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
                               add_ratio * tf.keras.layers.Conv2D(256, (1, 1), kernel_regularizer=l2(decay))(C3)])
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
                               add_ratio * tf.keras.layers.Conv2D(128, (1, 1), kernel_regularizer=l2(decay))(C2)])
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
                               add_ratio * tf.keras.layers.Conv2D(64, (1, 1), kernel_regularizer=l2(decay))(C1)])
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(2, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('sigmoid', name='AE')(x)
    return x


def decoding_concat_layer(x, C1, C2, C3, C4, decay, channel):
    add_ratio = 0.5
    #     x = tf.keras.layers.Conv2D(512, (1, 1))(x)  # (4, 4, 512)
    #     x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
    #                                add_ratio * tf.keras.layers.Conv2D(512, (1, 1), kernel_regularizer=l2(decay ))(C4)])

    x = tf.keras.layers.concatenate([C4, tf.keras.layers.UpSampling2D(size=(2, 2))(x)], axis=3)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)  # (8, 8, 512)
    #     x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
    #                                add_ratio * tf.keras.layers.Conv2D(256, (1, 1), kernel_regularizer=l2(decay ))(C3)])

    x = tf.keras.layers.concatenate([C3, tf.keras.layers.UpSampling2D(size=(2, 2))(x)], axis=3)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #     x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
    #                                add_ratio * tf.keras.layers.Conv2D(128, (1, 1), kernel_regularizer=l2(decay ))(C2)])

    x = tf.keras.layers.concatenate([C2, tf.keras.layers.UpSampling2D(size=(2, 2))(x)], axis=3)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #     x = tf.keras.layers.Add()([(1 - add_ratio) * tf.keras.layers.UpSampling2D(size=(2, 2))(x),
    #                                add_ratio * tf.keras.layers.Conv2D(64, (1, 1), kernel_regularizer=l2(decay ))(C1)])
    x = tf.keras.layers.concatenate([C1, tf.keras.layers.UpSampling2D(size=(2, 2))(x)], axis=3)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(decay))(x)
    x = tf.keras.layers.Conv2D(channel, (3, 3), padding='same', activation='sigmoid', kernel_regularizer=l2(decay),
                               name='unet_mobile_out')(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Activation('relu')(x)
    #     x = tf.keras.layers.Conv2D(3, (3, 3), padding='same', kernel_regularizer=l2(decay ))(x)
    #     x = tf.keras.layers.BatchNormalization()(x)
    #     x = tf.keras.layers.Activation('sigmoid', name='AE')(x)
    return x


def unet_mobile(decay=0, channel=2):
    input_Layer = tf.keras.layers.Input(shape=(512, 512, 1), name='unet_mobile_in')
    model_1, model_2, model_3, model_4, model_5 = encoding(pretrained_net='MobileNet')
    C1 = model_1(input_Layer)
    C2 = model_2(C1)
    C3 = model_3(C2)
    C4 = model_4(C3)
    C5 = model_5(C4)
    Out_Layer = decoding_concat_layer(C5, C1, C2, C3, C4, decay=decay, channel=channel)

    #     classifier = classification()
    #     Out_Layer_2 = classifier(C5)
    model = tf.keras.Model(inputs=[input_Layer], outputs=[Out_Layer])
    return model

