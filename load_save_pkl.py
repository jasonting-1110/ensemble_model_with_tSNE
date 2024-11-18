import os
import numpy as np
import random
import pickle
from datetime import datetime
from tensorflow import data
import tensorflow as tf

def save_data_pickle(data, save_dir, file_name):
    # 获取当前日期
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # 创建日期文件夹路径
    date_dir = os.path.join(save_dir, today_date)
    
    # 如果日期文件夹不存在，则创建它
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    
    # 完整文件路径
    file_path = os.path.join(date_dir, file_name)
    
    # 将数据保存到文件
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


#加入date參數，便能使用指定日期的pkl檔，而不會只限定於load當日pkl

 # # 获取当前日期
    # today_date = datetime.now().strftime("%Y-%m-%d")
    
    # # 创建日期文件夹路径
    # date_dir = os.path.join(save_dir, today_date)
def load_data_pickle(save_dir, file_name, date):
    # 如果没有指定日期，则使用当前日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 创建日期文件夹路径
    date_dir = os.path.join(save_dir, date)

    
    # 完整文件路径
    file_path = os.path.join(date_dir, file_name)
    
    # 从文件加载数据
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise FileNotFoundError(f"No file found at {file_path}")
    
    
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



def save_batches_to_memmap(generator, shape, dtype, filename):
        memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        idx = 0
        for batch in generator:
            batch_size = batch.shape[0]
            memmap_array[idx:idx+batch_size] = batch
            idx += batch_size
        return memmap_array

def load_npy_data(train_folders, test_folders, batch_size):
    cal_train_gen = npy_batch_data_generator(train_folders[1], batch_size)
    nor_train_gen = npy_batch_data_generator(train_folders[0], batch_size)
    cal_test_gen = npy_batch_data_generator(test_folders[1], batch_size)
    nor_test_gen = npy_batch_data_generator(test_folders[0], batch_size)
    return cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen

def fill_array_from_generator(generator, array):
    index = 0
    for batch in generator:
        batch_size = batch.shape[0]
        if index + batch_size > array.shape[0]:
            array[index:] = batch[:array.shape[0]-index]
            break
        array[index:index+batch_size] = batch
        index += batch_size

def generate_and_save_data(train_nor_path, train_cal_path, test_nor_path, test_cal_path, save_path):
    train_folders = [train_nor_path, train_cal_path]
    test_folders = [test_nor_path, test_cal_path]
    batch_size = 128

    cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen = load_npy_data(train_folders, test_folders, batch_size)

    cal_train_shape = (len(os.listdir(train_folders[1])), 224, 224, 3)
    nor_train_shape = (len(os.listdir(train_folders[0])), 224, 224, 3)
    cal_test_shape = (len(os.listdir(test_folders[1])), 224, 224, 3)
    nor_test_shape = (len(os.listdir(test_folders[0])), 224, 224, 3)

    cal_train_data = save_batches_to_memmap(cal_train_gen, cal_train_shape, np.float32, os.path.join(save_path, 'cal_train_data.memmap'))
    nor_train_data = save_batches_to_memmap(nor_train_gen, nor_train_shape, np.float32, os.path.join('nor_train_data.memmap'))
    cal_test_data = save_batches_to_memmap(cal_test_gen, cal_test_shape, np.float32, os.path.join('cal_test_data.memmap'))
    nor_test_data = save_batches_to_memmap(nor_test_gen, nor_test_shape, np.float32, os.path.join('nor_test_data.memmap'))

    train_data = np.memmap(os.path.join('train_data.memmap'), dtype=np.float32, mode='w+', shape=(cal_train_shape[0] + nor_train_shape[0], 224, 224, 3))
    test_data = np.memmap(os.path.join('test_data.memmap'), dtype=np.float32, mode='w+', shape=(cal_test_shape[0] + nor_test_shape[0], 224, 224, 3))


    train_data[:cal_train_shape[0]] = cal_train_data
    train_data[cal_train_shape[0]:] = nor_train_data
    test_data[:cal_test_shape[0]] = cal_test_data
    test_data[cal_test_shape[0]:] = nor_test_data

    cal_data = np.memmap(os.path.join('cal_data.memmap'), dtype=np.float32, mode='w+', shape=(cal_train_shape[0] + cal_test_shape[0], 224, 224, 3))
    nor_data = np.memmap(os.path.join('nor_data.memmap'), dtype=np.float32, mode='w+', shape=(nor_train_shape[0] + nor_test_shape[0], 224, 224, 3))


    cal_data[:cal_train_shape[0]] = cal_train_data
    cal_data[cal_train_shape[0]:] = cal_test_data
    nor_data[:nor_train_shape[0]] = nor_train_data
    nor_data[nor_train_shape[0]:] = nor_test_data


#有必要進行以下步驟處理生成好的train_data嗎?
    train_seq = np.arange(0, np.size(train_data, 0), 1)
    random.shuffle(train_seq)
    train_data = train_data[train_seq, :, :]

    nor_train_upto = len(nor_train_data)
    cal_train_upto = len(cal_train_data)

    cal_labels = np.ones((np.size(cal_data, 0), 1))
    nor_labels = np.zeros((np.size(nor_data, 0), 1))

    cal_train_labels, cal_test_labels = cal_labels[:cal_train_upto], cal_labels[cal_train_upto:]
    nor_train_labels, nor_test_labels = nor_labels[:nor_train_upto], nor_labels[nor_train_upto:]

    train_labels = np.vstack((cal_train_labels, nor_train_labels))
    test_labels = np.vstack((cal_test_labels, nor_test_labels))

    train_labels = train_labels[train_seq]

    print("Train labels shape:", train_labels.shape)
    print("Test labels shape:", test_labels.shape)
    
    file_name = "data.pkl"
    save_data_pickle((train_data,train_labels,test_data,test_labels), save_path,file_name) #可以考慮把train test拆分(因為tSNE不需要用到train，只會在訓練時用到)
    # save_data_pickle((train_data,train_labels), save_path, "train_" + file_name)
    # save_data_pickle((test_data,test_labels), save_path, "test_" +file_name)
#想辦法在save_data_pickle函式添加train_data,train_labels,這樣就可以跑完generate_and_save_data後直接訓練model



