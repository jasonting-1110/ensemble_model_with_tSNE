import os
import numpy as np
from multiprocessing.pool import ThreadPool

def load_batch(folder,file_list):
    batch_data = []
    for file in file_list:
        file_path = os.path.join(folder, file)
        img = np.load(file_path) 
        img = np.array(img)
        img = np.resize(img, (224, 224, 3))
        batch_data.append(img)
    return np.array(batch_data)

def npy_batch_data_generator(folder, batch_size):
    file_list = [file for file in os.listdir(folder) if not file.endswith('.db')]
    num_files = len(file_list)
    num_batches = num_files // batch_size
    pool = ThreadPool() #expects a function and an iterable of arguments, but you're passing three arguments instead ->error!
    for batch_idx in range(num_batches):
        batch_files = file_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]  #以batch為單位讀data
        #batch_data = pool.map(load_batch, [batch_files])[0]  #passing pool.map as list -> result in load_batch being called in 1 arg instead of 2
        #batch_data = pool.map(load_batch, [folder]*len(batch_files), batch_files)
        batch_data = pool.map(lambda file: load_batch(folder, file), batch_files)
        #yield batch_data
        yield np.array(batch_data)
    pool.close()
    pool.join()

def load_npy_data(train_folders, test_folders, batch_size):
    cal_train_gen = npy_batch_data_generator(train_folders[0], batch_size)
    nor_train_gen = npy_batch_data_generator(train_folders[1], batch_size)
    cal_test_gen = npy_batch_data_generator(test_folders[0], batch_size)
    nor_test_gen = npy_batch_data_generator(test_folders[1], batch_size)
    return cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen