import os
import cv2
import numpy as np
import argparse

def process_image(image_path, target_dim):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    target_height, target_width = target_dim

    inconsistent = False

    if height != target_height or width != target_width:
        inconsistent = True

    if height > target_height:
        image = image[:target_height, :]
    elif height < target_height:
        padding = np.zeros((target_height - height, width, 3), dtype=np.uint8)
        image = np.vstack((image, padding))

    if width > target_width:
        image = image[:, :target_width]
    elif width < target_width:
        padding = np.zeros((target_height, target_width - width, 3), dtype=np.uint8)
        image = np.hstack((image, padding))

    return image, inconsistent

def fix_image_dimensions(input_folder, output_folder, target_dim):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    inconsistent_count = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            processed_image, inconsistent = process_image(input_path, target_dim)
            if inconsistent:
                inconsistent_count += 1
            cv2.imwrite(output_path, processed_image)

    print(f"Number of images with inconsistent dimensions: {inconsistent_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix image dimensions.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')
    parser.add_argument('target_height', type=int, help='Target height of the images')
    parser.add_argument('target_width', type=int, help='Target width of the images')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    target_dim = (args.target_height, args.target_width)

    fix_image_dimensions(input_folder, output_folder, target_dim)