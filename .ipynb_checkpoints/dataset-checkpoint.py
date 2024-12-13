import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image  # Import PIL for conversion to PIL.Image
from torchvision import transforms

# Define transforms for training and validation
img_size = 224  # Desired image size
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def read_xray(path_file):
    """
    Reads a chest X-ray image from the given path and converts it to 3 channels.
    """
    xray = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    if xray is None:
        raise ValueError(f"Error reading image at path: {path_file}")
    xray = cv2.cvtColor(xray, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    return xray


class PneumoniaDataset(Dataset):
    """
    Dataset class for Pneumonia detection.
    Applies transformations for training and validation datasets.
    Test dataset remains unchanged.
    """
    def __init__(self, dataset_df, transform=None, test_mode=False):
        """
        Args:
            dataset_df (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Transformations to apply to the images.
            test_mode (bool): If True, no transformations are applied.
        """
        self.dataset = dataset_df
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample (image and its label) at the given index.
        
        Args:
            idx (int): Index of the data sample.
        
        Returns:
            dict: A dictionary containing the image (`img`), label (`target`), and image name (`Name`).
        """
        img_path = self.dataset['Path'].iloc[idx]
        img = read_xray(img_path)  # Read the image (numpy or PIL.Image)
        label = int(self.dataset['Label'].iloc[idx])  # Ensure the label is an integer
        name = self.dataset['Name'].iloc[idx]
    
        # Convert NumPy array to PIL.Image for torchvision transforms
        img = Image.fromarray(img)
    
        # Always apply the transformation
        if self.transform:
            img = self.transform(img)
    
        return {
            'Name': name,
            'img': img,
            'target': label
        }

