

import os,re
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


"""
# # 指定目錄路徑
# directory = r"D:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\2024_8_13\cal"

# # 獲取目錄中的子資料夾列表
# subfolders = sorted( [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))] )

# #python的數字排序比較特殊
# # for i, subfolder in enumerate(subfolders):
# #     print(f"{i + 1}: {subfolder}")

# # 按子資料夾名稱中的數字部分進行排序
# subfolders_sorted = sorted(subfolders, key=lambda x: int(x[1:]))

# # 列出所有排序後的子資料夾名稱
# print("All sorted subfolders:")
# for i, subfolder in enumerate(subfolders_sorted):
#     print(f"{i + 1}: {subfolder}")

# # 遍歷排序後前10個子資料夾
# print("\nProcessing first 10 sorted subfolders:")
# for subfolder in subfolders_sorted[:10]:
#     subfolder_path = os.path.join(directory, subfolder)
#     print(f"Processing folder: {subfolder_path}")
#     # 在此處添加對每個子資料夾中資料的處理邏輯"""

# def copy_nor(src_base_path,dest_path):

#     # # 定义源和目标路径
#     # src_base_path = r'\\BOIL-NAS\homes\311514061\2024-5haveBG-SD-OCT dental calculus\2024dentalCalculus'
#     # dest_path = r'D:\OCT\dental OCT\bare tooth\ensemble_model_aug\code\2024_dentalCalculus\NOR'

#     # 创建目标目录（如果不存在）
#     os.makedirs(dest_path, exist_ok=True)

#     # src_base_path = r'\\BOIL-NAS\homes\311514061\2024-5haveBG-SD-OCT dental calculus\2024dentalCalculus' : 參考此路徑的folder思考語法微調
    
#     #用os.listdir中遍歷前10個

#     for i in range(1, 8):
#         src_folder = os.path.join(src_base_path, f'NOR{i}', 'npy_resize', 'npy')  #可微調前綴   
#         if os.path.exists(src_folder):
#             print(f'正在复制 {src_folder} 中的文件到 {dest_path}')
            
#             for root, dirs, files in os.walk(src_folder):
#                 for file in files:
#                     src_file = os.path.join(root, file)
#                     # 使用源文件夹名作为前缀，避免因為檔案相同而被覆蓋掉
#                     dest_file = os.path.join(dest_path, f'NOR{i}_{file}')   #可微調前綴
#                     # 检查目标文件是否存在
#                     if not os.path.exists(dest_file):
#                         shutil.copy2(src_file, dest_file)
#                     else:
#                         # 添加后缀以避免覆盖
#                         base, ext = os.path.splitext(dest_file)
#                         copy_index = 1
#                         new_dest_file = f"{base}_{copy_index}{ext}"
#                         while os.path.exists(new_dest_file):
#                             copy_index += 1
#                             new_dest_file = f"{base}_{copy_index}{ext}"
#                         shutil.copy2(src_file, new_dest_file)
#         else:
#             print(f'{src_folder} 不存在')

#     print('完成复制任务')

def copy_nor(src_base_path,dest_path):
    # 定義正則表達式來匹配以 n 開頭並跟隨一個或多個數字的資料夾名稱:n1 n2 n3 n4...
    pattern = re.compile(r'^n\d+$') 

    # 找出符合正則表達式的子資料夾名稱
    matching_folders = [f for f in os.listdir(src_base_path) if os.path.isdir(os.path.join(src_base_path, f)) and pattern.match(f)]


    # 按子資料夾名稱中的數字部分進行排序: 因為python的listdir排序不完全依照數字大小排 ex:14 15 2 25 3 37...
    subfolders_sorted = sorted(matching_folders, key=lambda x: int(x[1:]))

    for subfolder in subfolders_sorted:
        src_folder = os.path.join(src_base_path, f'{subfolder}', 'npy_resize', 'npy')  #可微調前綴
        if os.path.exists(src_folder):
            print(f'正在复制 {src_folder} 中的文件到 {dest_path}')

            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    src_file = os.path.join(root, file)
                    # 使用源文件夹名作为前缀，避免因為檔案相同而被覆蓋掉
                    dest_file = os.path.join(dest_path, f'{subfolder}_{file}')   #可微調前綴
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




def copy_cal(src_base_path,dest_path):
    # 定義正則表達式來匹配以以c開頭並跟隨一個或多個數字的資料夾名:c6 c8 c14...
    pattern = re.compile(r'^c\d+$') 

    # 找出符合正則表達式的子資料夾名稱
    matching_folders = [f for f in os.listdir(src_base_path) if os.path.isdir(os.path.join(src_base_path, f)) and pattern.match(f)]


    #可以考慮改名!
    # # 擷取數字並將 c 改為 cal
    # renamed_folders = [f"cal{pattern.match(folder).group(1)}" for folder in matching_folders]

    # # 列出修改後的資料夾名稱
    # for folder in renamed_folders:
    #     print(folder)


    # 按子資料夾名稱中的數字部分進行排序: 因為python的listdir排序不完全依照數字大小排 ex:14 15 2 25 3 37...
    subfolders_sorted = sorted(matching_folders, key=lambda x: int(x[1:]))

    for subfolder in subfolders_sorted:
        src_folder = os.path.join(src_base_path, f'{subfolder}', 'npy_resize', 'npy')  #可微調前綴
        if os.path.exists(src_folder):
            print(f'正在复制 {src_folder} 中的文件到 {dest_path}')

            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    src_file = os.path.join(root, file)
                    # 使用源文件夹名作为前缀，避免因為檔案相同而被覆蓋掉
                    dest_file = os.path.join(dest_path, f'{subfolder}_{file}')   #可微調前綴
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

    # 遍历 supra1_v 到 supra8_v 目录
    for i in range(8, 9):
        src_folder = os.path.join(src_base_path, f'supra{i}_h', 'npy_resize', 'npy') #可微調前綴
        
        if os.path.exists(src_folder):
            print(f'正在复制 {src_folder} 中的文件到 {dest_path}')
            
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    src_file = os.path.join(root, file)
                    # 使用源文件夹名作为前缀，避免因為檔案相同而被覆蓋掉
                    dest_file = os.path.join(dest_path, f'supra{i}_h_{file}') #可微調前綴
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



