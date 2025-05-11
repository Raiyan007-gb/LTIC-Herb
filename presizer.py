import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class PreSizer:
    def __init__(self, target_resolution=(512, 512), crop_size=448, boundary_width=20):
        self.target_resolution = target_resolution
        self.crop_size = crop_size
        self.boundary_width = boundary_width

    def __call__(self, image):
        # Remove boundary pixels
        image = image[self.boundary_width:-self.boundary_width, self.boundary_width:-self.boundary_width, :]

        # Pad to make image resolution symmetric
        height, width = image.shape[:2]
        max_dim = max(height, width)
        top = (max_dim - height) // 2
        bottom = max_dim - height - top
        left = (max_dim - width) // 2
        right = max_dim - width - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)

        # Resize the image to target resolution
        image = cv2.resize(image, self.target_resolution)

        # Perform central cropping
        crop_top = (self.target_resolution[0] - self.crop_size) // 2
        crop_left = (self.target_resolution[1] - self.crop_size) // 2
        image = image[crop_top:crop_top + self.crop_size, crop_left:crop_left + self.crop_size, :]

        return image

# Define dataset class
class HerbariumDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for dirpath, _, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_list.append(os.path.join(dirpath, filename))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = self.file_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, image_path

# Define transformation pipeline
pre_sizer = transforms.Compose([
    PreSizer(),
    transforms.ToTensor(),  # Convert to PyTorch tensor
])

# Define dataset
herbarium_dataset = HerbariumDataset(root_folder='./train_image', transform=pre_sizer)
#print(len(herbarium_dataset))

# Create DataLoader
data_loader = DataLoader(herbarium_dataset, batch_size=4, shuffle=True)

# Output directory for saving images
output_dir = './Herb_presizer_data'  # Change this to your desired output directory

# Save images with original directory structure
for batch in data_loader:
    images, paths = batch
    for img, path in zip(images, paths):
        subfolder = os.path.dirname(path.replace('./train_image', output_dir))
        os.makedirs(subfolder, exist_ok=True)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Convert to numpy array
        cv2.imwrite(os.path.join(output_dir, path.replace('./train_image', output_dir)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
