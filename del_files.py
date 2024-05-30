#有bug 就restart kernel
import os
import shutil

def delete_files_in_directory(*directories): # * 可以一次引入多個目錄
    for directory in directories: #要外加for loop，如果要一次刪多個目錄
        # 遍历目录中的所有文件和子目录
        for root, dirs, files in os.walk(directory):
            # 删除所有文件
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")
            
            # 递归删除子目录中的所有文件
            for dir in dirs:
                subdir_path = os.path.join(root, dir)
                delete_files_in_directory(subdir_path)



####################################################################
def delete_folder(folder_path):
    """
    刪除指定的資料夾及其所有子資料夾和文件。

    :param folder_path: 要刪除的資料夾的路徑
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"資料夾 '{folder_path}' 已刪除。")
    else:
        print(f"資料夾 '{folder_path}' 不存在。")

