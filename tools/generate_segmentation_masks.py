from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import numpy as np
import torch
import tqdm as tqdm
import optparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# parse passed image folders
optparser = optparse.OptionParser()
optparser.add_option("-i", "--image_folder", dest="image_folder", help="Image folder to process")
optparser.add_option("-o", "--output_folder", dest="output_folder", help="Output folder for masks")

image_folder =  optparser.parse_args()[0].image_folder
output_folder =  optparser.parse_args()[0].output_folder

if not image_folder:
    print("Please provide an image folder to process")
    exit()

if not os.path.exists(image_folder):
    print(f"Image folder {image_folder} does not exist")
    exit()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f'Processing images in {image_folder}')
for image_file in tqdm.tqdm(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_file)

    mask_image_path = image_path.replace(image_folder, output_folder)
    print(f'Processing {mask_image_path}')
    if not os.path.exists(mask_image_path):
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            try:
                masks = mask_generator.generate(image)
            except Exception as e:
                print(e)
                print(f'Error processing {image_path}')
                continue
            processed_image = np.zeros_like(image)

            for i, mask in enumerate(masks):
                color = (i%255, i*10%255, i*100%255)
                color_mask = np.zeros_like(image)
                color_mask[ : , : , : ] = color
                segmentation_mask = np.array(mask['segmentation'], dtype=np.uint8)
                processed_image += cv2.bitwise_and(color_mask, color_mask, mask=segmentation_mask)

            cv2.imwrite(mask_image_path, processed_image)
