import os
import numpy as np
from PIL import Image
import random
import pickle
from datetime import datetime

def npy_batch_data_generator(folder, batch_size):
    file_list = [file for file in os.listdir(folder) if not file.endswith('.db')] #去除thumb.db的因子
    num_files = len(file_list)
    num_batches = num_files // batch_size
    for batch_idx in range(num_batches):
        batch_files = file_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_data = []
        for file in batch_files:
            file_path = os.path.join(folder, file)
            img = np.load(file_path) 
            img = np.array(img)
            img = np.resize(img, (224, 224, 3))
            batch_data.append(img)
        yield np.array(batch_data)
def load_npy_data(train_folders, test_folders, batch_size):
    cal_train_gen = npy_batch_data_generator(train_folders[0], batch_size)
    nor_train_gen = npy_batch_data_generator(train_folders[1], batch_size)
    cal_test_gen = npy_batch_data_generator(test_folders[0], batch_size)
    nor_test_gen = npy_batch_data_generator(test_folders[1], batch_size)
    return cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen

def img_batch_data_generator(folder, batch_size):
    file_list = [file for file in os.listdir(folder) if not file.endswith('.db')] #去除thumb.db的因子
    num_files = len(file_list)
    num_batches = num_files // batch_size
    for batch_idx in range(num_batches):
        batch_files = file_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_data = []
        for file in batch_files:
            file_path = os.path.join(folder, file)
            img = Image.open(file_path)  
            img = np.array(img)
            img = np.resize(img, (224, 224, 3))
            batch_data.append(img)
        yield np.array(batch_data)

def load_img_data(train_folders, test_folders, batch_size):
    cal_train_gen = img_batch_data_generator(train_folders[0], batch_size)
    nor_train_gen = img_batch_data_generator(train_folders[1], batch_size)
    cal_test_gen = img_batch_data_generator(test_folders[0], batch_size)
    nor_test_gen = img_batch_data_generator(test_folders[1], batch_size)
    return cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen

def save_data_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def generate_and_save_data(train_nor_path,train_cal_path,test_nor_path,test_cal_path,save_path):
    train_folders = [train_nor_path, train_cal_path]
    test_folders = [test_nor_path, test_cal_path]
    batch_size = 128
    
    cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen = load_npy_data(train_folders, test_folders, batch_size)

    #創建內存映射文件有效地避免内存不足導致電腦當機
    def save_batches_to_memmap(generator, shape, dtype, filename):
        memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        idx = 0
        for batch in generator:
            batch_size = batch.shape[0]
            memmap_array[idx:idx+batch_size] = batch
            idx += batch_size
        return memmap_array

    cal_train_shape = (len(os.listdir(train_folders[0])), 224, 224, 3)
    nor_train_shape = (len(os.listdir(train_folders[1])), 224, 224, 3)
    cal_test_shape = (len(os.listdir(test_folders[0])), 224, 224, 3)
    nor_test_shape = (len(os.listdir(test_folders[1])), 224, 224, 3)

    cal_train_data = save_batches_to_memmap(cal_train_gen, cal_train_shape, np.float32, 'cal_train_data.memmap')
    nor_train_data = save_batches_to_memmap(nor_train_gen, nor_train_shape, np.float32, 'nor_train_data.memmap')
    cal_test_data = save_batches_to_memmap(cal_test_gen, cal_test_shape, np.float32, 'cal_test_data.memmap')
    nor_test_data = save_batches_to_memmap(nor_test_gen, nor_test_shape, np.float32, 'nor_test_data.memmap')

    train_data = np.memmap('train_data.memmap', dtype=np.float32, mode='w+', shape=(cal_train_shape[0] + nor_train_shape[0], 224, 224, 3))
    test_data = np.memmap('test_data.memmap', dtype=np.float32, mode='w+', shape=(cal_test_shape[0] + nor_test_shape[0], 224, 224, 3))

    train_data[:cal_train_shape[0]] = cal_train_data
    train_data[cal_train_shape[0]:] = nor_train_data
    test_data[:cal_test_shape[0]] = cal_test_data
    test_data[cal_test_shape[0]:] = nor_test_data

    cal_data = np.memmap('cal_data.memmap', dtype=np.float32, mode='w+', shape=(cal_train_shape[0] + cal_test_shape[0], 224, 224, 3))
    nor_data = np.memmap('nor_data.memmap', dtype=np.float32, mode='w+', shape=(nor_train_shape[0] + nor_test_shape[0], 224, 224, 3))

    cal_data[:cal_train_shape[0]] = cal_train_data
    cal_data[cal_train_shape[0]:] = cal_test_data
    nor_data[:nor_train_shape[0]] = nor_train_data
    nor_data[nor_train_shape[0]:] = nor_test_data

    train_seq = (np.arange(0, np.size(train_data, 0), 1))
    random.shuffle(train_seq)
    train_data = train_data[train_seq, :, :]

    #用len函式，就不用每次都要自己調參
    nor_train_upto = len(nor_train_data)
    cal_train_upto = len(cal_train_data)

    cal_labels = np.ones((np.size(cal_data, 0), 1))
    nor_labels = np.zeros((np.size(nor_data, 0), 1))

    cal_train_labels, cal_test_labels = cal_labels[:cal_train_upto], cal_labels[cal_train_upto:]
    nor_train_labels, nor_test_labels = nor_labels[:nor_train_upto], nor_labels[nor_train_upto:]

    train_labels = np.vstack((cal_train_labels, nor_train_labels))
    test_labels = np.vstack((cal_test_labels, nor_test_labels))

    train_labels = train_labels[train_seq] # out of bound error

    print("Train labels shape:", train_labels.shape)
    print("Test labels shape:", test_labels.shape)

    save_data_pickle((test_data, test_labels), save_path) #成功存取pkl後，memmap就可del:針對你想儲存的變量動態增減save_data_pickle的參數
    # save_data_pickle((test_data, test_labels), save_path,file_name)

        

    