# pickle_utils.py
import os
import pickle
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def define_pickle_paths(experiment_name):
    """
    Define paths for real and fake pickle files based on the experiment name in the folder experiments_pickle_files.
    """
    # Define the folder path
    folder_path = "experiments_pickle_files"
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Create paths for the pickle files within the folder
    real_pickle_path = os.path.join(folder_path, f"train_real_{experiment_name}.pickle")
    fake_pickle_path = os.path.join(folder_path, f"train_fake_{experiment_name}.pickle")
    
    return real_pickle_path, fake_pickle_path

def get_image_paths_from_folder(data_dir):
    """
    Get all image paths from a directory and its subdirectories.
    """
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def create_pickle_file(image_paths, sample_size, output_pickle_path, seed=None):
    """
    Create a pickle file with a specified number of randomly sampled image paths.
    """
    if seed is not None:
        random.seed(seed)
    if sample_size is not None:
        image_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
    
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(image_paths, f)

def create_pickle_files_if_needed(opt):
    """
    Check if pickle files exist and create them if needed using parameters from the parser arguments.
    """
    # Define pickle paths using opt.experiment_name
    real_pickle_path, fake_pickle_path = define_pickle_paths(opt.experiment_name)

    # Create pickle file for real images if it doesn't exist
    if not os.path.exists(real_pickle_path):
        real_image_paths = get_image_paths_from_folder(opt.real_path)
        create_pickle_file(real_image_paths, opt.real_sample_size, real_pickle_path, seed=opt.seed)
        print(f"Created pickle file for real images: {real_pickle_path}")
    else:
        print(f"Real pickle file already exists: {real_pickle_path}")

    # Create pickle file for fake images if it doesn't exist
    if not os.path.exists(fake_pickle_path):
        combined_fake_image_paths = []
        for path, sample_size in zip(opt.fake_paths, opt.fake_sample_sizes):
            fake_image_paths = get_image_paths_from_folder(path)
            
            # If sample_size is None, use all images from the current fake path
            if sample_size is None:
                sampled_fake_image_paths = fake_image_paths
            else:
                sampled_fake_image_paths = random.sample(fake_image_paths, min(sample_size, len(fake_image_paths)))
            
            combined_fake_image_paths.extend(sampled_fake_image_paths)
        
        # Save the combined fake image paths to a single pickle file
        create_pickle_file(combined_fake_image_paths, None, fake_pickle_path, seed=opt.seed)
        print(f"Created pickle file for fake images: {fake_pickle_path}")
    else:
        print(f"Fake pickle file already exists: {fake_pickle_path}")

    return real_pickle_path, fake_pickle_path

