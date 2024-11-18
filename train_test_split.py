import os

import shutil


#因為我可能要一次傳入多個Nor或Cal，使用*prefixes搭配for loop，這樣就不用一次定義多個prefix參數
def split(src_path,train_path,test_path,*prefixes): 
    # 确保目标目录存在，否则创建
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    # 遍历源目录中的文件
    for filename in os.listdir(src_path):
        copy_to_test = False
        for prefix in prefixes:
            if filename.startswith(prefix):  # 如果文件以指定的前缀开头
                # 复制到测试目录
                shutil.copy(os.path.join(src_path, filename), test_path)
                copy_to_test = True
                break
                
        if not copy_to_test:
            # 复制到训练目录
            shutil.copy(os.path.join(src_path, filename), train_path)

    print("Files copied successfully.")

