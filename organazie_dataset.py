import os
from shutil import move

def organize_files(base_dir):
    # Define the directory paths
    depth_dir = os.path.join(base_dir, 'depth_images')
    segmentation_dir = os.path.join(base_dir, 'segmentation_images')
    clean_dir = os.path.join(base_dir, 'clean_images')

    # Create directories if they don't exist
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # Move files to respective directories
    for file in os.listdir(base_dir):
        if file.endswith('depth.png'):
            move(os.path.join(base_dir, file), os.path.join(depth_dir, file))
        elif file.endswith('segmentation.png'):
            move(os.path.join(base_dir, file), os.path.join(segmentation_dir, file))
        elif file.endswith('clean.png'):
            move(os.path.join(base_dir, file), os.path.join(clean_dir, file))

if __name__ == '__main__':
    # Specify the base directory where the files are initially located
    base_directory = '/home/work/roomMaker/clean_images'
    organize_files(base_directory)
