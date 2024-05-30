import os
import numpy as np
import multiprocessing

def npy_batch_data_loader(queue, folder, batch_size):
    file_list = [file for file in os.listdir(folder) if not file.endswith('.db')]
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
        queue.put(np.array(batch_data))

def npy_batch_data_generator(folder, batch_size):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=npy_batch_data_loader, args=(queue, folder, batch_size))
    process.start()
    while True:
        batch_data = queue.get()
        yield batch_data
        process.join(0.1)  # 控制等待的时间，避免死锁

def load_npy_data(train_folders, test_folders, batch_size):
    cal_train_gen = npy_batch_data_generator(train_folders[0], batch_size)
    nor_train_gen = npy_batch_data_generator(train_folders[1], batch_size)
    cal_test_gen = npy_batch_data_generator(test_folders[0], batch_size)
    nor_test_gen = npy_batch_data_generator(test_folders[1], batch_size)
    return cal_train_gen, nor_train_gen, cal_test_gen, nor_test_gen




