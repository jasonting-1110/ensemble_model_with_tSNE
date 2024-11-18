import os
def create_directories(root_directory):
    # Create the root directory if it doesn't exist
    os.makedirs(root_directory, exist_ok=True)

    # Define the folder names
    train_folder = "train_FOR_YOUR_LIFE"
    test_folder = "test_FOR_YOUR_LIFE"

    # Create the train and test directories
    train_directory = os.path.join(root_directory, train_folder)
    test_directory = os.path.join(root_directory, test_folder)

    # Create the train and test directories if they don't exist
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Create subdirectories within train and test for cal and nor
    cal_train_directory = os.path.join(train_directory, "cal")
    nor_train_directory = os.path.join(train_directory, "nor")
    cal_test_directory = os.path.join(test_directory, "cal")
    nor_test_directory = os.path.join(test_directory, "nor")

    # Create the subdirectories
    os.makedirs(cal_train_directory, exist_ok=True)
    os.makedirs(nor_train_directory, exist_ok=True)
    os.makedirs(cal_test_directory, exist_ok=True)
    os.makedirs(nor_test_directory, exist_ok=True)

    print("Directories created successfully.")


#create_directories(r"D:\OCT\shit")
    
