from train_unet import *
if __name__ == '__main__' :
    file = 'exp'
    model_list = ['unet']
    batch_size_list = [4]
    lr_list = [1e-4]
    focal_p_list = [0.7]
    decay_list = [0]
    max_epoch = 100
    interval = 20
    train = True
    initial_exp = 0
    start_exp = 0
    device_number = 0

    grid_search(file, model_list, batch_size_list, lr_list, focal_p_list, decay_list, max_epoch, interval, train, initial_exp, start_exp, device_number)
