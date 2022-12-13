from train_unet_cv import *
import os
device_num = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = device_num #// 초기화할 GPU number
if __name__ == '__main__' :

    df_path = 'DATA/meta_train.csv'
    img_dir = 'DATA/train/x_512'
    img_test_dir = 'DATA/test/x'
    mask_dir = 'DATA/train/y_512'

    file = 'cv/exp_test_test_2'
    model_list = ['dl3_mobilenetv2', 'dl3_xception']#, 'transunet']# ['unet', 'vnet', 'reunet']  # ['unet']#, 'unet_mobile'] 'transunet'
    batch_size_list = [2]#, 4, 8]
    cv_k_list = [0]  # , 1, 2, 3]
    # , 4, 8, 16]#, 8, 4]#, 2, 1]#[1,2,4,8]
    lr_list = [1e-4]#, 1e-5]  # , 1e-5]#, 1e-5]#, 1e-5]#, 1e-5]
    focal_p_list = [0.5]  # , 0.3]#, 0.6, 0.75]#, 0.60, 0.65, 0.75, 0.9]
    decay_list = [0]
    max_epoch = 3
    interval = 1
    initial_exp = 0
    start_exp = 0
    train = True

    my_gs = grid_search(df_path, img_dir, mask_dir, img_test_dir, file, cv_k_list, model_list, batch_size_list,
                        lr_list, focal_p_list, decay_list, max_epoch, interval, train, initial_exp, start_exp,
                        device_num)
