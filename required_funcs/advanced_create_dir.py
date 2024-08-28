import os
def create_directories(root_directory):
    # Create the root directory if it doesn't exist
    os.makedirs(root_directory, exist_ok=True)

    # Define the folder names
    NOR1_folder = "NOR1"
    NOR2_folder = "NOR2"
    NOR3_folder = "NOR3"
    NOR4_folder = "NOR4"
    NOR5_folder = "NOR5"
    NOR6_folder = "NOR6"
    NOR7_folder = "NOR7"
    NOR8_folder = "NOR8"
    # Create the train and test directories
    NOR1_directory = os.path.join(root_directory, NOR1_folder)
    NOR2_directory = os.path.join(root_directory, NOR2_folder)
    NOR3_directory = os.path.join(root_directory, NOR3_folder)
    NOR4_directory = os.path.join(root_directory, NOR4_folder)
    NOR5_directory = os.path.join(root_directory, NOR5_folder)
    NOR6_directory = os.path.join(root_directory, NOR6_folder)
    NOR7_directory = os.path.join(root_directory, NOR7_folder)
    NOR8_directory = os.path.join(root_directory, NOR8_folder)

    # Create the train and test directories if they don't exist
    os.makedirs(NOR1_directory, exist_ok=True)
    os.makedirs(NOR2_directory, exist_ok=True)
    os.makedirs(NOR3_directory, exist_ok=True)
    os.makedirs(NOR4_directory, exist_ok=True)
    os.makedirs(NOR5_directory, exist_ok=True)
    os.makedirs(NOR6_directory, exist_ok=True)
    os.makedirs(NOR7_directory, exist_ok=True)
    os.makedirs(NOR8_directory, exist_ok=True)
    

    # Create train and test for nor : divide Nor folders data into 8:2 & assemble them in these 2 folders
    nor_train_directory = os.path.join(root_directory, "train")
    nor_test_directory = os.path.join(root_directory, "test")

    # Create the subdirectories
    os.makedirs(nor_train_directory, exist_ok=True)
    os.makedirs(nor_test_directory, exist_ok=True)
    

    print("Directories created successfully.")


#create_directories(r"D:\OCT\shit")
    
