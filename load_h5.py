import os
import numpy as np
import random
import pickle
from datetime import datetime
import tensorflow as tf

import h5py

#若pkl會有內存問題，可以考慮使用hdf5/zarr存取data!

#加入date參數，便能使用指定日期的pkl檔，而不會只限定於load當日pkl


#之後可以想辦法簡化資料生成流程，因為太過繁瑣!!


  
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

#把generator的資料填入array
def fill_array_from_generator(generator, array):
    index = 0
    for batch in generator:
        batch_size = batch.shape[0]
        if index + batch_size > array.shape[0]: #如果batch_size > array大小，該batch不能完全放入array，只能填入一部份
            array[index:] = batch[:array.shape[0]-index]  
            break
        array[index:index+batch_size] = batch #如果 batch 可以完全填入 array 中，則將 batch 的資料填入 array 的對應位置
        index += batch_size  #定位下一個批次應填入的位置



def data_generator(data, labels, batch_size):
    num_samples = data.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        yield data[start_idx:end_idx], labels[start_idx:end_idx]


def save_data_hdf5(data_generator, save_dir, file_name, batch_size):
    today_date = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(save_dir, today_date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    file_path = os.path.join(date_dir, file_name)

    # 使用生成器獲取第一批數據，以確定數據形狀
    first_batch_data, first_batch_labels = next(data_generator)
    num_samples = first_batch_data.shape[0]
    data_shape = first_batch_data.shape[1:]
    labels_shape = first_batch_labels.shape[1:]

    # 創建 HDF5 文件並初始化數據集
    with h5py.File(file_path, 'w') as f:
        data_dataset = f.create_dataset('data', (0,) + data_shape, maxshape=(None,) + data_shape, dtype='float32', compression="gzip")
        labels_dataset = f.create_dataset('labels', (0,) + labels_shape, maxshape=(None,) + labels_shape, dtype='float32', compression="gzip")

        # Write the first batch
        data_dataset.resize((num_samples,) + data_shape)
        labels_dataset.resize((num_samples,) + labels_shape)
        data_dataset[:] = first_batch_data
        labels_dataset[:] = first_batch_labels

        # Append remaining batches
        total_written_samples = num_samples
        for batch_data, batch_labels in data_generator:
            batch_size = batch_data.shape[0]
            total_written_samples += batch_size
            
            # Resize datasets
            data_dataset.resize((total_written_samples,) + data_shape)
            labels_dataset.resize((total_written_samples,) + labels_shape)

            # Write the batch data
            data_dataset[-batch_size:] = batch_data
            labels_dataset[-batch_size:] = batch_labels

        print(f"Data successfully saved to {file_path}")


# 在 generate_and_save_data 中使用 data_generator

#疑問，我發現該函式已經有做完label跟資料隨機化了，是否能夠再執行完此函式後，直接train model?
def generate_and_save_data(train_nor_path, train_cal_path, test_nor_path, test_cal_path, save_path):
    train_folders = [train_nor_path, train_cal_path]
    test_folders = [test_nor_path, test_cal_path]
    batch_size = 128

    # 加載訓練和測試生成器(批次讀取)
    cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen = load_npy_data(train_folders, test_folders, batch_size)

    cal_train_shape = (len(os.listdir(train_folders[1])), 224, 224, 3)
    nor_train_shape = (len(os.listdir(train_folders[0])), 224, 224, 3)
    cal_test_shape = (len(os.listdir(test_folders[1])), 224, 224, 3)
    nor_test_shape = (len(os.listdir(test_folders[0])), 224, 224, 3)

    cal_train_data = np.zeros(cal_train_shape, dtype=np.float32)
    nor_train_data = np.zeros(nor_train_shape, dtype=np.float32)
    cal_test_data = np.zeros(cal_test_shape, dtype=np.float32)
    nor_test_data = np.zeros(nor_test_shape, dtype=np.float32)

    fill_array_from_generator(cal_train_gen, cal_train_data)
    fill_array_from_generator(nor_train_gen, nor_train_data)
    fill_array_from_generator(cal_test_gen, cal_test_data)
    fill_array_from_generator(nor_test_gen, nor_test_data)

    train_data = np.concatenate([cal_train_data, nor_train_data], axis=0)
    test_data = np.concatenate([cal_test_data, nor_test_data], axis=0)

    #binary labels，one-hot-encoding通常用於多類別分類
    cal_train_labels = np.ones((cal_train_shape[0], 1))
    nor_train_labels = np.zeros((nor_train_shape[0], 1))
    cal_test_labels = np.ones((cal_test_shape[0], 1))
    nor_test_labels = np.zeros((nor_test_shape[0], 1))

    train_labels = np.concatenate([cal_train_labels, nor_train_labels], axis=0)
    test_labels = np.concatenate([cal_test_labels, nor_test_labels], axis=0)

    #已經有隨機化處理，但還是overfit!
    train_seq = np.arange(train_data.shape[0])
    np.random.shuffle(train_seq)
    train_data = train_data[train_seq]
    train_labels = train_labels[train_seq]

    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    print("Train labels shape:", train_labels.shape)
    print("Test labels shape:", test_labels.shape)

    # 創建生成器
    train_gen = data_generator(train_data, train_labels, batch_size)
    test_gen = data_generator(test_data, test_labels, batch_size)

    # 保存為 HDF5 文件
    save_data_hdf5(train_gen, save_path, "train_data.h5", batch_size)
    save_data_hdf5(test_gen, save_path, "test_data.h5", batch_size)
    
    return train_data,test_data,train_labels,test_labels  #明天試看看是否有回傳這些變數!





class HDF5DataLoader: #可能是batch size太大造成gpu內存不足!
    def __init__(self, save_dir,file_name, date=None, batch_size=1000, data_key='data', labels_key='labels'):

        # self.file_path = file_path
        self.save_dir = save_dir
        self.file_name = file_name
        self.batch_size = batch_size
        self.data_key = data_key
        self.labels_key = labels_key

        # 如果沒有指定日期，則使用當前日期
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # 創建日期文件夾路徑
        self.date_dir = os.path.join(save_dir, date)
        self.file_path = os.path.join(self.date_dir, file_name)
        
        # 確保文件存在
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在。")
        
        with h5py.File(self.file_path, 'r') as f:
             self.total_samples = f[self.data_key].shape[0]  #num_of_samples :第一維度
            # self.total_samples =len(f[self.data_key]) # integer (此行也適用)
            # self.total_samples = f[self.data_key][:] # numpy array
           
            # self.total_samples =[f[self.key][:] for self.key in f.keys()]
    
    def __iter__(self):
        with h5py.File(self.file_path, 'r') as f:
            if self.data_key not in f or self.labels_key not in f:
                raise KeyError(f"無法在 HDF5 文件中找到 '{self.data_key}' 或 '{self.labels_key}' 數據集。")
            
            for i in range(0, self.total_samples, self.batch_size):
                end_index = min(i + self.batch_size, self.total_samples)
                batch_data = f[self.data_key][i:end_index]
                batch_labels = f[self.labels_key][i:end_index]
                yield batch_data, batch_labels
    
    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size


    

#一次讀取整個data(會占內存!)
def load_data_hdf5(save_dir, file_name, date=None):
    # 如果没有指定日期，则使用当前日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 创建日期文件夹路径
    date_dir = os.path.join(save_dir, date)
    
    # 完整文件路径
    file_path = os.path.join(date_dir, file_name)
    
    # 从 HDF5 文件加载数据
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            data = [f[key][:] for key in f.keys()]
        return data
    else:
        raise FileNotFoundError(f"No file found at {file_path}")


