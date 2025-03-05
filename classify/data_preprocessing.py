import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader  # Import DataLoader
from PIL import Image

# Custom padding function to make images square
def pad_image(image, target_size=(1280, 1280)):
    h, w = image.size
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2
    padded_image = Image.new(image.mode, target_size, (0, 0, 0))  # Black padding
    padded_image.paste(image, (pad_w, pad_h))
    return padded_image

# Define augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 144)),  # Resize to a smaller resolution
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
def load_dataset(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

# Split dataset into train, validation, and test sets
def split_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset

# Create data loaders
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=8):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader