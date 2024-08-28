import os
import numpy as np
import random
import pickle
from datetime import datetime
import joblib
import h5py

#若pkl會有內存問題，可以考慮使用hdf5/zarr存取data!
# def save_data_pickle(data, save_dir, file_name):
#     # 获取当前日期
#     today_date = datetime.now().strftime("%Y-%m-%d")
    
#     # 创建日期文件夹路径
#     date_dir = os.path.join(save_dir, today_date)
    
#     # 如果日期文件夹不存在，则创建它
#     if not os.path.exists(date_dir):
#         os.makedirs(date_dir)
    
#     # 完整文件路径
#     file_path = os.path.join(date_dir, file_name)
    
#     # 将数据保存到文件
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)


# #加入date參數，便能使用指定日期的pkl檔，而不會只限定於load當日pkl
# def load_data_pickle(save_dir, file_name, date = None):
    # 如果没有指定日期，则使用当前日期
    # if date is None:
    #     date = datetime.now().strftime("%Y-%m-%d")
    
    # # 创建日期文件夹路径
    # date_dir = os.path.join(save_dir, date)

    
    # # 完整文件路径
    # file_path = os.path.join(date_dir, file_name)
    
    # # 从文件加载数据
    # if os.path.exists(file_path):
    #     # with open(file_path, 'rb') as file:
    #     #     while True: #用此分塊加載數據
    #     #         try:
    #     #             data = pickle.load(file)
    #     #             # 处理数据
    #     #         except EOFError:
    #     #             break
    #     #         return data

    #     # 使用 joblib 加载数据，并启用内存映射，應付大型檔案內存占用問題所導致的環境失靈
    #     data = joblib.load(file_path, mmap_mode='r')
    #     return data
    # else:
        # raise FileNotFoundError(f"No file found at {file_path}")
    
def save_data_hdf5(data, save_dir, file_name):
    # 获取当前日期
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # 创建日期文件夹路径
    date_dir = os.path.join(save_dir, today_date)
    
    # 如果日期文件夹不存在，则创建它
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    
    # 完整文件路径
    file_path = os.path.join(date_dir, file_name)
    
    # 将数据保存到 HDF5 文件
    with h5py.File(file_path, 'w') as f:
        for i, d in enumerate(data):
            f.create_dataset(f'dataset_{i}', data=d, compression="gzip")

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

    
def load_data_hdf5_in_batches(save_dir, file_name, date=None, batch_size=1000): #總共有5911筆，拆為5 batches!
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    date_dir = os.path.join(save_dir, date)
    file_path = os.path.join(date_dir, file_name)
    
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as f:
            # 獲取所有鍵
            keys = list(f.keys())
            
            # 確保至少有兩個鍵
            if len(keys) < 2:
                raise ValueError("The HDF5 file must contain at least two datasets.")
            
            # 假設前兩個鍵分別對應數據和標籤
            data_key = keys[0]
            labels_key = keys[1]
            
            data = f[data_key]
            labels = f[labels_key]
            
            # 檢查數據和標籤的大小是否匹配
            assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples"
            
            for i in range(0, data.shape[0], batch_size):
                yield data[i:i+batch_size], labels[i:i+batch_size]
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
def load_npy_data(train_folders, test_folders, batch_size):
    cal_train_gen = npy_batch_data_generator(train_folders[0], batch_size)
    nor_train_gen = npy_batch_data_generator(train_folders[1], batch_size)
    cal_test_gen = npy_batch_data_generator(test_folders[0], batch_size)
    nor_test_gen = npy_batch_data_generator(test_folders[1], batch_size)
    return cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen


def generate_and_save_data(train_nor_path, train_cal_path, test_nor_path, test_cal_path, save_path):
    train_folders = [train_nor_path, train_cal_path]
    test_folders = [test_nor_path, test_cal_path]
    batch_size = 128

    cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen = load_npy_data(train_folders, test_folders, batch_size)

    cal_train_shape = (len(os.listdir(train_folders[1])), 224, 224, 3)
    nor_train_shape = (len(os.listdir(train_folders[0])), 224, 224, 3)
    cal_test_shape = (len(os.listdir(test_folders[1])), 224, 224, 3)
    nor_test_shape = (len(os.listdir(test_folders[0])), 224, 224, 3)

    # 使用 numpy 數組而不是 memmap
    cal_train_data = np.zeros(cal_train_shape, dtype=np.float32)
    nor_train_data = np.zeros(nor_train_shape, dtype=np.float32)
    cal_test_data = np.zeros(cal_test_shape, dtype=np.float32)
    nor_test_data = np.zeros(nor_test_shape, dtype=np.float32)

    # 填充數據
    fill_array_from_generator(cal_train_gen, cal_train_data)
    fill_array_from_generator(nor_train_gen, nor_train_data)
    fill_array_from_generator(cal_test_gen, cal_test_data)
    fill_array_from_generator(nor_test_gen, nor_test_data)

    # 合併訓練和測試數據
    train_data = np.concatenate([cal_train_data, nor_train_data], axis=0)
    test_data = np.concatenate([cal_test_data, nor_test_data], axis=0)

    # 準備標籤
    cal_train_labels = np.ones((cal_train_shape[0], 1))
    nor_train_labels = np.zeros((nor_train_shape[0], 1))
    cal_test_labels = np.ones((cal_test_shape[0], 1))
    nor_test_labels = np.zeros((nor_test_shape[0], 1))

    train_labels = np.concatenate([cal_train_labels, nor_train_labels], axis=0)
    test_labels = np.concatenate([cal_test_labels, nor_test_labels], axis=0)

    # 打亂訓練數據
    train_seq = np.arange(train_data.shape[0])
    np.random.shuffle(train_seq)
    train_data = train_data[train_seq]
    train_labels = train_labels[train_seq]

    print("Train labels shape:", train_labels.shape)
    print("Test labels shape:", test_labels.shape)

    # # 保存數據
    # file_name = "data.pkl"
    # save_data_pickle((train_data, train_labels, test_data, test_labels), save_path, file_name)

    # train test分開保存
    # train_file_name = "train_data.pkl"
    # test_file_name = "test_data.pkl"

    # save_data_pickle((train_data, train_labels), save_path, train_file_name)
    # save_data_pickle((test_data, test_labels), save_path, test_file_name)

    # train test分开保存为 HDF5 文件
    train_file_name = "train_data.h5"
    test_file_name = "test_data.h5"

    save_data_hdf5((train_data, train_labels), save_path, train_file_name)
    save_data_hdf5((test_data, test_labels), save_path, test_file_name)

def fill_array_from_generator(generator, array):
    index = 0
    for batch in generator:
        batch_size = batch.shape[0]
        if index + batch_size > array.shape[0]:
            array[index:] = batch[:array.shape[0]-index]
            break
        array[index:index+batch_size] = batch
        index += batch_size