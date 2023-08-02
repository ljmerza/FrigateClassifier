import os
import shutil
import random

# Define the path to the original dog_images folder
data_dir = './dog_images'
# Define the path to the new dog_images folder
gen_data = 'dog_images_formatted'

# Define the paths for the training, validation, and test data folders
train_dir = f'./{gen_data}/train'
validation_dir = f'./{gen_data}/validation'
test_dir = f'./{gen_data}/test'

# Function to remove directories and their content
def remove_dirs(dir_paths):
    for dir_path in dir_paths:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

# Remove existing directories and their content
remove_dirs([train_dir, validation_dir, test_dir])

# Create the directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all breed folders
breed_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Set the random seed for reproducibility
random.seed(42)

# Function to convert breed folder name to capital case
def format_breed_name(breed_folder):
    return breed_folder.split('-')[-1].replace('_', ' ').title()

# Iterate over each breed folder and split the images into train, validation, and test sets
for breed_folder in breed_folders:
    breed_path = os.path.join(data_dir, breed_folder)
    image_files = os.listdir(breed_path)
    
    # Shuffle the image files
    random.shuffle(image_files)
    
    # Calculate the split sizes
    num_images = len(image_files)
    num_train = int(0.8 * num_images)
    num_validation = int(0.1 * num_images)
    
    # Split the image files into train, validation, and test sets
    train_files = image_files[:num_train]
    validation_files = image_files[num_train:num_train + num_validation]
    test_files = image_files[num_train + num_validation:]
    
    # Get the formatted breed name
    breed_name = format_breed_name(breed_folder)
    
    # Create the breed folders in the train, validation, and test directories
    os.makedirs(os.path.join(train_dir, breed_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, breed_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, breed_name), exist_ok=True)
    
    # Move the images to the corresponding directories
    for file in train_files:
        src_path = os.path.join(breed_path, file)
        dest_path = os.path.join(train_dir, breed_name, file)
        shutil.copy(src_path, dest_path)
    
    for file in validation_files:
        src_path = os.path.join(breed_path, file)
        dest_path = os.path.join(validation_dir, breed_name, file)
        shutil.copy(src_path, dest_path)
    
    for file in test_files:
        src_path = os.path.join(breed_path, file)
        dest_path = os.path.join(test_dir, breed_name, file)
        shutil.copy(src_path, dest_path)
