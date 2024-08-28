from PIL import Image #只適用圖檔!
import os
import numpy as np

def convert_npy_to_png(src_folder, png_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        npy_path = os.path.join(src_folder, filename)
        # 确保文件是 .npy 格式且存在
        if filename.endswith('.npy') and os.path.isfile(npy_path):
            # 构建目标 PNG 文件路径
            png_path = os.path.join(png_folder, os.path.splitext(filename)[0] + '.png')
            if not os.path.exists(png_path):
                try:
                    # 载入 .npy 文件
                    img_array = np.load(npy_path)
                    
                    # 确保数组数据是 8 位无符号整数（常见图像格式）
                    if img_array.dtype != np.uint8:
                        img_array = img_array.astype(np.uint8)
                    
                    # 将 NumPy 数组转换为 PIL 图像
                    img = Image.fromarray(img_array)
                    
                    # 保存图像为 PNG 格式
                    img.save(png_path)
                    print(f'Converted {filename} to PNG format.')
                except Exception as e:
                    print(f'Failed to convert {filename}: {e}')
            else:
                print(f'{png_path} already exists.')

                
def cvt_to_png(src_folder,png_folder):
    # 如果PNG文件夾不存在，則創建它
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 遍歷源文件夾中的每個子文件夾
    for subdir in os.listdir(src_folder):
        subdir_path = os.path.join(src_folder, subdir)
        # 確保子文件夾存在且是目錄
        if os.path.isdir(subdir_path):
            # 在PNG文件夾中創建與子文件夾對應的文件夾
            png_subdir = os.path.join(png_folder, subdir)
            if not os.path.exists(png_subdir):
                os.makedirs(png_subdir)
            # 遍歷子文件夾中的文件
            for filename in os.listdir(subdir_path):
                npy_path = os.path.join(subdir_path, filename)
                # 確保文件是某格式且存在(我的是npy)
                if filename.endswith('.npy') and os.path.isfile(npy_path):
                    # 確保目標PNG文件不存在
                    png_path = os.path.join(png_subdir, os.path.splitext(filename)[0] + '.png')
                    if not os.path.exists(png_path):
                        
                        # 載入文件並將其轉換為PNG格式
                        #with np.load(npy_path) as img_array:   #np.load不能與with共用!

                        # 确保数组数据是8位无符号整数（常见图像格式）
                        img_array = np.load(npy_path)
                        if img_array.dtype != np.uint8:
                            img_array = img_array.astype(np.uint8)

                        # 将 NumPy 数组转换为 PIL 图像
                        img = Image.fromarray(img_array)

                        # 保存图像为 PNG 格式
                        img.save(png_path)
                        print(f'Converted {filename} to PNG format.')
                    else:
                        print(f'{png_path} already exists.')
