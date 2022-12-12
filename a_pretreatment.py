import pandas as pd
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import shutil
from tqdm import tqdm


def load_dataset(df, k, random_state, reset_k = False) :
    df['class'] = 0
    if reset_k == True :
        folds = StratifiedKFold(n_splits=k, random_state=random_state, shuffle=True)
        df['kfold'] = -1
        for i in range(k):
            df_idx, valid_idx = list(folds.split(df.values, df['class']))[i]
            valid = df.iloc[valid_idx]
            df.loc[df[df.index.isin(valid.index) == True].index.to_list(), 'kfold'] = i
        #df.to_csv('Data/train_k.csv')
    else :
        pass
    return df

def resize_img(img, target_shape, save_dir, save_name, save_format = 'png') :
    os.makedirs(save_dir, exist_ok = True)
    if save_format != 'png' :
        save_name = save_name.split('.')[0] + '.' + save_format
    save_path = os.path.join(save_dir, save_name)
    img = cv2.resize(img, (target_shape, target_shape), interpolation = cv2.INTER_AREA)
#     plt.imshow(img)
#     plt.show()
    cv2.imwrite(save_path, img)



def make_df() :
    for img_fname, mask_fname in tqdm(zip(img_list, mask_list)) :
        my_dict = {}
        mask = cv2.imread(mask_fname, cv2.IMREAD_GRAYSCALE)
        #print(mask.max())
        img = cv2.imread(img_fname)

        resize_img(img, target_shape, target_img_root, os.path.basename(img_fname), save_format = 'jpg')
        resize_img(mask, target_shape, target_mask_root, os.path.basename(mask_fname)) 
        width = mask.shape[1]
        height = mask.shape[0]
    #     mask[mask==225] = 255
    #     mask[mask!=255] = 0
    #     mask = 255 - mask
    #     print(np.unique(mask))
        #cv2.imwrite(os.path.join(mask_mod_root, os.path.basename(mask_fname)), mask)
        my_dict['img_fname'] = os.path.basename(mask_fname).split('.')[0] + '.jpg'
        my_dict['mask_fname'] = os.path.basename(mask_fname)
        my_dict['width'] = width
        my_dict['height'] = height
        my_dict_list.append(my_dict)


        #fname = os.path.basename(mask_fname)
        #origin = os.path.join(img_root, fname)
        #target = os.path.join(target_img_root, fname)
        #shutil.copy(origin, target)

    df = pd.DataFrame(my_dict_list)
    df_k = load_dataset(df, k= 4, random_state = 11, reset_k = True)
    df_test = df_k[df_k['kfold'] == 0]
    df_train = df_k[df_k['kfold'] != 0]
    df_test.to_csv('DATA/meta_test.csv')#, index = False)
    df_train.to_csv('DATA/meta_train.csv')#, index = False)    


    #     plt.imshow(mask, cmap = 'gray')
    #     plt.show()
if __name__ == '__main__' : 
    mask_root = 'DATA/train/y'
    img_root = 'DATA/train/x'
    my_dict_list = []
    mask_list = sorted(glob.glob(os.path.join(mask_root, '*.png')))
    img_list = sorted(glob.glob(os.path.join(img_root, '*.png')))
    target_shape = 512
    target_img_root = 'DATA/train/x_{}'.format(target_shape)
    target_mask_root = 'DATA/train/y_{}'.format(target_shape)
    os.makedirs(target_img_root, exist_ok = True)
    os.makedirs(target_mask_root, exist_ok = True)
    make_df()