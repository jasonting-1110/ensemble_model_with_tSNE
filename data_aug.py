# import os
# import cv2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # 設定原始檔案路徑和目的檔案路徑
# src_paths = [r'D:\OCT\dental OCT\bare tooth\ensemble_model\train\cal',r'D:\OCT\dental OCT\bare tooth\ensemble_model\train\nor']
# des_paths = [r'D:\OCT\dental OCT\bare tooth\ensemble_model_aug\train\cal',r'D:\OCT\dental OCT\bare tooth\ensemble_model_aug\train\nor'] #增加到30393張

# # 設定資料增強的參數 :ImageDataGenerator 預期4d vector
# #調參試看看
# datagen = ImageDataGenerator(  
#     rotation_range=5, # rotation_range=k->angle : -k~k
#     width_shift_range=0,  # movement : ratio of total width
#     height_shift_range=0, # movement : ratio of total height
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# #print(os.listdir(src_path))

# # 讀取原始資料並進行資料增強 : 一個一個print! 
# for src_path, des_path in zip(src_paths, des_paths):
#     for subdir in os.listdir(src_path):  # ok
#         print(subdir)
#         src_file = os.path.join(src_path, subdir)
#         print(src_file)


#         # 檢查檔案是否為圖像檔案
#         if os.path.isfile(src_file):
#             # 讀取圖像檔案
#             image = cv2.imread(src_file)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 必須先轉換為RGB色彩空間
#             #print(image.shape) #(224,224,3)
            
#             # 將圖像轉換為4D張量 (samples, height, width, channels)
#             image = image.reshape((1,) + image.shape)
#             #print(image.shape) #(1,224,224,3)


            
#             # 進行資料增強，並儲存增強後的圖像
#             i = 0
#             #先不要改名!跟原始圖像區隔
#             for batch in datagen.flow(image, batch_size=1, save_to_dir=des_path, save_prefix='augmented_', save_format='tif'):  
#                 i += 1
#                 if i >= 5:  # 假設每張原始圖像產生5張增強後的圖像
#                     break  # 確保不會無窮迴圈

# print("完成資料增強並儲存至目的檔案路徑。")

#####################################
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_augmentation_img(src_paths, des_paths):
    # 設定資料增強的參數 :ImageDataGenerator 預期4d vector
    #調參試看看
    datagen = ImageDataGenerator(  
        rotation_range=5, # rotation_range=k->angle : -k~k
        width_shift_range=0,  # movement : ratio of total width
        height_shift_range=0, # movement : ratio of total height
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

   # # 讀取原始資料並進行資料增強 : 一個一個print! 
    for src_path, des_path in zip(src_paths, des_paths):
        for subdir in os.listdir(src_path):  # ok
            #print(subdir)
            src_file = os.path.join(src_path, subdir)
            #print(src_file)


            # 檢查檔案是否為圖像檔案
            if os.path.isfile(src_file):
                # 讀取圖像檔案
                image = cv2.imread(src_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 必須先轉換為RGB色彩空間
                #print(image.shape) #(224,224,3)
                
                # 將圖像轉換為4D張量 (samples, height, width, channels)
                image = image.reshape((1,) + image.shape)
                #print(image.shape) #(1,224,224,3)


                
                # 進行資料增強，並儲存增強後的圖像
                i = 0
                #先不要改名!跟原始圖像區隔
                for batch in datagen.flow(image, batch_size=1, save_to_dir=des_path, save_prefix='augmented_', save_format='tif'):  
                    i += 1
                    if i >= 2:  # 假設每張原始圖像產生2張增強後的圖像
                        break  # 確保不會無窮迴圈

    print("完成資料增強並儲存至目的檔案路徑。")

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def data_augmentation_npy(src_paths, des_paths):
    # 设置数据增强的参数 :ImageDataGenerator 预期 4D 张量
    # 调参试看看
    datagen = ImageDataGenerator(  
        rotation_range=5,  # rotation_range=k->angle : -k~k
        width_shift_range=0,  # movement : ratio of total width
        height_shift_range=0,  # movement : ratio of total height
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # 读取原始数据并进行数据增强
    for src_path, des_path in zip(src_paths, des_paths):
        for subdir in os.listdir(src_path):  #顯示src_path底下的所有檔案
            src_file = os.path.join(src_path, subdir) #用.join建立路徑，以免出現PermissionError

            # 检查文件是否为文件夹
            #if os.path.isdir(src_file):
            if os.path.isfile(src_file):
                # 遍历文件夹中的 Numpy 数组文件
                #for file in os.listdir(src_file):
                    # npy_path = os.path.join(src_file, file)
                    
                    # # 检查文件是否为 Numpy 数组文件
                    # if os.path.isfile(npy_path) and npy_path.endswith('.npy'):
                    #     # 加载 Numpy 数组文件
                    #     array = np.load(npy_path)
                    # 检查文件是否为 Numpy 数组文件
                    if os.path.isfile(src_file) and src_file.endswith('.npy'):
                        # 加载 Numpy 数组文件
                        array = np.load(src_file)
                        
                        # 将 Numpy 数组转换为 4D 张量 (samples, height, width, channels)->axis要選好!
                        array = np.expand_dims(array, axis=-1)  # 扩展为 4D 张量（单通道灰度图像）
                        array = array.reshape((1,) + array.shape)
                        
                        # 进行数据增强，并保存增强后的图像
                        i = 0
                        #注意datagen.flow期望input為4d tensor(可能需要添加擴充維度的code)
                        #由於ImageDataGenerator 类不支持直接保存 .npy
                        #for batch in datagen.flow(array, batch_size=1, save_to_dir=des_path, save_prefix='augmented_' + subdir + '_', save_format='npy'):
                        #由於ImageDataGenerator 类不支持直接保存 .npy，先用save_format='jpeg'再搭配np.save
                        for batch in datagen.flow(array, batch_size=1, save_to_dir=des_path, save_prefix='augmented_' + subdir + '_', save_format='jpeg'):
                            augmented_array = batch[0]
                            # 保存增强后的 Numpy 数组为 .npy 文件
                            np.save(os.path.join(des_path, 'augmented_' + subdir + '_' + str(i) + '.npy'), augmented_array)
                            i += 1
                            if i >= 2:  # 假设每个原始数组产生 2 个增强后的数组
                                break  # 确保不会无限循环

    print("完成数据增强并保存至目标文件路径。")

    



