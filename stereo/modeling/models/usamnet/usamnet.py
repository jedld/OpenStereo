import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from .transforms import transform_disparity_fn, transform_fn, transform_seg_fn, test_transform_fn, test_transform_seg_fn
import cv2
import pdb

import torch.nn as nn

class BaselineStereoCNN(nn.Module):
    def __init__(self, device):
        super(BaselineStereoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255
        
    
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        left_img = transform(left_img).unsqueeze(0).to(self.device)
        right_img = transform(right_img).unsqueeze(0).to(self.device)
        input = torch.cat((left_img, right_img), 1)
        return self.forward(input), None
    
class BaselineStereoCNN2(nn.Module):
    def __init__(self, device, kitti=False):
        super(BaselineStereoCNN2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        if kitti:
            output_padding = 1
            output_padding2 = (0, 1)
        else:
            output_padding = 0
            output_padding2 = 0

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=output_padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1, output_padding=output_padding2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)

        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255
        
        
    

    
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        left_img = transform(left_img).unsqueeze(0).to(self.device)
        right_img = transform(right_img).unsqueeze(0).to(self.device)
        input = torch.cat((left_img, right_img), 1)
        return self.forward(input), None
    
class BaseSegmentationCNN(nn.Module):
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        transform_segmentation = test_transform_seg_fn()

        # Remove the code that removes the batch dimension
        # left_img and right_img are tensors of shape (batch_size, channels, height, width)

        # Generate masks for each image in the batch
        left_mask = self.generate_segment_map(left_img)

        # Apply transforms to each image in the batch
        left_img = torch.stack([transform(img) for img in left_img])
        right_img = torch.stack([transform(img) for img in right_img])
        left_mask = torch.stack([transform_segmentation(mask) for mask in left_mask])

        # Move tensors to the appropriate device
        left_img = left_img.to(self.device)
        right_img = right_img.to(self.device)
        left_mask = left_mask.to(self.device)

        # Concatenate along the channel dimension
        input = torch.cat((left_img, right_img, left_mask), dim=1)
        return self.forward(input), left_mask


    def generate_segment_map(self, images):
        # images: tensor of shape (batch_size, channels, height, width)
        processed_images = []

        for img in images:
            # Convert tensor to numpy array and rearrange dimensions to (height, width, channels)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            masks = self.mask_generator.generate(img_np)
            processed_img = np.zeros_like(img_np)

            for i, mask in enumerate(masks):
                color = (i % 255, i * 10 % 255, i * 100 % 255)
                color_mask = np.zeros_like(img_np)
                color_mask[:, :, :] = color
                segmentation_mask = np.array(mask['segmentation'], dtype=np.uint8)
                processed_img += cv2.bitwise_and(color_mask, color_mask, mask=segmentation_mask)

            # Convert the processed image back to a tensor and rearrange dimensions to (channels, height, width)
            processed_img_tensor = torch.from_numpy(processed_img).permute(2, 0, 1)
            processed_images.append(processed_img_tensor)

        # Stack all processed images to form a batch
        processed_images = torch.stack(processed_images)
        return processed_images

class SegStereoCNN2(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False, kitti=False):
        super(SegStereoCNN2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        if kitti:
            output_padding = 1
            output_padding2 = (0, 1)
        else:
            output_padding = 0
            output_padding2 = 0

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=output_padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1, output_padding=output_padding2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255

class SegStereoCNN(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SegStereoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out + x

class SASegStereoCNN(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SASegStereoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        
        self.self_attention = SelfAttention(1024)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255


class SASegStereoCNN2(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False, kitti=False):
        super(SASegStereoCNN2, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        self.self_attention = SelfAttention(1024)

        if kitti:
            output_padding = 1
            output_padding2 = (0, 1)
        else:
            output_padding = 0
            output_padding2 = 0

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=output_padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1, output_padding=output_padding2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255


class SAStereoCNN2(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False, kitti=False, full_res=False):
        super(SAStereoCNN2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        self.self_attention = SelfAttention(1024)
        if kitti:
            output_padding = 1
            output_padding2 = (0, 1)
        elif full_res:
            output_padding = (1, 1)
            output_padding2 = 0
        else:
            output_padding = 0
            output_padding2 = 0

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=output_padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1, output_padding=output_padding2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)

        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255

class SAStereoCNN3(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False, kitti=False):
        super(SAStereoCNN3, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        self.self_attention = SelfAttention(1024)
        if kitti:
            output_padding = 1
            output_padding2 = (0, 1)
        else:
            output_padding = 0
            output_padding2 = 0

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=output_padding),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1, output_padding=output_padding2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.pathway = nn.Sequential(
            nn.ConvTranspose2d(6, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = torch.concat((self.up5(up4),self.pathway(x)),1)
        return self.conv(up5) * 255
class SaUNet(SegStereoCNN2):
    def __init__(self, cfgs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(device, True, kitti=cfgs.KITTI)
        self.max_disp = cfgs.MAX_DISP
        if cfgs.LOAD_PRETRAIN:
            print("Loading pretrained model")
            self.load_state_dict(torch.load('pretrained/seg-unet.checkpoint'))
        # self.load_state_dict(torch.load('stereo_cnn_stereo_cnn_sa_baseline.checkpoint'))

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        segmentation = data['seg']

        input = torch.cat((image1, image2, segmentation), 1)
        result = super().forward(input)

        return { 'disp_pred': result }

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"].unsqueeze(1)  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        disp_pred = model_pred['disp_pred']
        loss = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask])
        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info

class SaUSAMNet(SASegStereoCNN2):
    def __init__(self, cfgs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(device, True, kitti=cfgs.KITTI)
        self.max_disp = cfgs.MAX_DISP
        if cfgs.LOAD_PRETRAIN:
            print("Loading pretrained model")
            self.load_state_dict(torch.load('pretrained/seg-usamnet.checkpoint'))
        # self.load_state_dict(torch.load('stereo_cnn_stereo_cnn_sa_baseline.checkpoint'))

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        segmentation = data['seg']

        input = torch.cat((image1, image2, segmentation), 1)
        result = super().forward(input)

        return { 'disp_pred': result }

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"].unsqueeze(1)  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        disp_pred = model_pred['disp_pred']
        loss = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask])
        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info

class USAMNet(SAStereoCNN2):
    def __init__(self, cfgs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(device, kitti=cfgs.KITTI)
        self.max_disp = cfgs.MAX_DISP
        if cfgs.LOAD_PRETRAIN:
            print("Loading pretrained model")
            self.load_state_dict(torch.load('pretrained/usamnet.checkpoint'))

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        input = torch.cat((image1, image2), 1)
        result = super().forward(input)
        return { 'disp_pred': result }

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"].unsqueeze(1)  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        disp_pred = model_pred['disp_pred']
        loss = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask])
        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info

class USAMNetv2(SAStereoCNN3):
    def __init__(self, cfgs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(device, kitti=cfgs.KITTI)
        self.max_disp = cfgs.MAX_DISP
        if cfgs.LOAD_PRETRAIN:
            print("Loading pretrained model")
            self.load_state_dict(torch.load('pretrained/usamnet.checkpoint'))

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        input = torch.cat((image1, image2), 1)
        result = super().forward(input)
        return { 'disp_pred': result }

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"].unsqueeze(1)  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        disp_pred = model_pred['disp_pred']
        loss = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask])
        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info

class UNet(BaselineStereoCNN2):
    def __init__(self, cfgs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(device, cfgs.KITTI)
        self.max_disp = cfgs.MAX_DISP
        if cfgs.LOAD_PRETRAIN:
            print("Loading pretrained model")
            self.load_state_dict(torch.load('pretrained/unet.checkpoint'))

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        input = torch.cat((image1, image2), 1)
        result = super().forward(input)
        return { 'disp_pred': result }

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"].unsqueeze(1)  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        disp_pred = model_pred['disp_pred']
        loss = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask])
        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info