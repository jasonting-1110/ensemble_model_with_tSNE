# import os
# import shutil
# def copy_file(src_paths,des_paths):
# # Set source paths and destination paths
# # src_paths = [
# #     r'D:\OCT\dental OCT\bare tooth\ensemble_model\test\cal',
# #     r'D:\OCT\dental OCT\bare tooth\ensemble_model\test\nor'
# # ]
# # des_paths = [
# #     r'D:\OCT\dental OCT\bare tooth\ensemble_model_aug\test\cal',
# #     r'D:\OCT\dental OCT\bare tooth\ensemble_model_aug\test\nor'
# # ]

#     # Copy images from source to destination
#     for src_path, des_path in zip(src_paths, des_paths):
#         # Create destination directory if it doesn't exist
#         os.makedirs(des_path, exist_ok=True)
        
#         # Get list of image files in source directory
#         image_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        
#         # Copy each image file to the destination directory
#         for image_file in image_files:
#             src_file = os.path.join(src_path, image_file)
#             des_file = os.path.join(des_path, image_file)
#             shutil.copyfile(src_file, des_file)

#     print("Images copied successfully!")

import os
import shutil

def copy_npy_files(source_dirs, dest_base_dir):
    """
    將指定目錄中的.npy文件複製到目標目錄中
    """
    for source_dir in source_dirs:
        # 獲取目標子目錄名稱 (NOR1, NOR2, ..., NOR8)
        nor_subdir = os.path.basename(os.path.dirname(source_dir)) 
        #os.path.dirname: 去掉最後一個檔案名稱npy
        #os.path.basename: 返回去掉npy後，最後一個檔案名稱npy_resize


        # 確保目標子目錄存在
        dest_dir = os.path.join(dest_base_dir, nor_subdir)
        os.makedirs(dest_dir, exist_ok=True)

        # 列出所有以 'NOR' 開頭的文件
        for filename in os.listdir(source_dir):
            if filename.startswith('NOR') and filename.endswith('.npy'):
                source_file = os.path.join(source_dir, filename)
                dest_file = os.path.join(dest_dir, filename)
                
                # 複製文件
                shutil.copyfile(source_file, dest_file) #原本寫.copy2
                print(f"Copied {source_file} to {dest_file}")

if __name__ == "__main__":
    # 指定來源目錄
    base_source_dir = r"\\BOIL-NAS\homes\311514061\2024-5haveBG-SD-OCT dental calculus\2024dentalCalculus"
    #source_dirs = [os.path.join(base_source_dir, f'NOR{i}\npy_resize\npy') for i in range(1, 8)]  #\n換行語法導致混淆!
    source_dirs = [os.path.join(base_source_dir, f'NOR{i}','npy_resize','npy') for i in range(1, 8)]

    # 指定目標目錄
    dest_base_dir = r"D:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\NOR"
    dest_dirs = [os.path.join(dest_base_dir, f'NOR{i}') for i in range(1, 8)]

    # 執行複製操作
    copy_npy_files(source_dirs, dest_base_dir)



def copy_nor(src_base_path,dest_path):

    # 定义源和目标路径
    src_base_path = r'\\BOIL-NAS\homes\311514061\2024-5haveBG-SD-OCT dental calculus\2024dentalCalculus'
    dest_path = r'D:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\2024_dentalCalculus\NOR'

    # 创建目标目录（如果不存在）
    os.makedirs(dest_path, exist_ok=True)

    # 遍历 NOR1 到 NOR7 目录
    for i in range(1, 8):
        src_folder = os.path.join(src_base_path, f'NOR{i}', 'npy_resize', 'npy')
        
        if os.path.exists(src_folder):
            print(f'正在复制 {src_folder} 中的文件到 {dest_path}')
            
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    src_file = os.path.join(root, file)
                    # 使用源文件夹名作为前缀，避免因為檔案相同而被覆蓋掉
                    dest_file = os.path.join(dest_path, f'NOR{i}_{file}')
                    # 检查目标文件是否存在
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
                    else:
                        # 添加后缀以避免覆盖
                        base, ext = os.path.splitext(dest_file)
                        copy_index = 1
                        new_dest_file = f"{base}_{copy_index}{ext}"
                        while os.path.exists(new_dest_file):
                            copy_index += 1
                            new_dest_file = f"{base}_{copy_index}{ext}"
                        shutil.copy2(src_file, new_dest_file)
        else:
            print(f'{src_folder} 不存在')

    print('完成复制任务')




def copy_supra(src_base_path,dest_path):
    # 创建目标目录（如果不存在）
    os.makedirs(dest_path, exist_ok=True)

    # 遍历 supra6 到 supra8 目录
    for i in range(6, 9):
        src_folder = os.path.join(src_base_path, f'supra{i}', 'npy_resize', 'npy')
        
        if os.path.exists(src_folder):
            print(f'正在复制 {src_folder} 中的文件到 {dest_path}')
            
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    src_file = os.path.join(root, file)
                    # 使用源文件夹名作为前缀，避免因為檔案相同而被覆蓋掉
                    dest_file = os.path.join(dest_path, f'supra{i}_{file}')
                    # 检查目标文件是否存在
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
                    else:
                        # 添加后缀以避免覆盖
                        base, ext = os.path.splitext(dest_file)
                        copy_index = 1
                        new_dest_file = f"{base}_{copy_index}{ext}"
                        while os.path.exists(new_dest_file):
                            copy_index += 1
                            new_dest_file = f"{base}_{copy_index}{ext}"
                        shutil.copy2(src_file, new_dest_file)
        else:
            print(f'{src_folder} 不存在')

    print('完成复制任务')

import random
def train_test_split_copy(src_path,train_dest_path,test_dest_path):
    # 获取源目录中的文件列表
    files = os.listdir(src_path)

    # 打乱文件列表顺序
    random.shuffle(files)

    # 计算训练集和测试集的分割点
    train_proportion = 0.8 
    split_index = int(len(files) * train_proportion)

    # 复制文件到训练集目录
    for file in files[:split_index]:
        src_file = os.path.join(src_path, file)
        dest_file = os.path.join(train_dest_path, file)
        shutil.copy2(src_file, dest_file)

    # 复制文件到测试集目录
    for file in files[split_index:]:
        src_file = os.path.join(src_path, file)
        dest_file = os.path.join(test_dest_path, file)
        shutil.copy2(src_file, dest_file)

    print('完成复制任务')



