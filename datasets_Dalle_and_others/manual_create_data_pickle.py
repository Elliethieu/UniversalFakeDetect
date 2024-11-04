import os
import pickle

def get_image_paths_from_folders(base_dir):
    """
    Recursively get all image paths from all subdirectories inside the base_dir.
    """
    image_paths = set()  # Using a set to avoid duplicates
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file is an image by extension
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                # Append the full path of the image
                image_paths.add(os.path.join(root, file))  # Adding to a set
    return list(image_paths)  # Convert back to a list before returning

def save_paths_to_pickle(paths, output_pickle_path):
    """
    Save the list of image paths to a pickle file.
    """
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(paths, f)

# Paths to Dalle and others image folders (assuming they are in the same directory as the script)
Dalle_image_base_dir = os.path.join(os.getcwd(), "Dalle")
others_image_base_dir = os.path.join(os.getcwd(), "others")

# Get the paths for Dalle and others images
Dalle_image_paths = get_image_paths_from_folders(Dalle_image_base_dir)
others_image_paths = get_image_paths_from_folders(others_image_base_dir)

# Save the paths to pickle files
save_paths_to_pickle(Dalle_image_paths,  os.path.join(os.path.dirname(__file__), "Dalle", "train.pickle"))
save_paths_to_pickle(others_image_paths,  os.path.join(os.path.dirname(__file__), "others", "train.pickle"))

print(f"Dalle images: {len(Dalle_image_paths)}")
print(f"others images: {len(others_image_paths)}")
