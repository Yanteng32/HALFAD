import gc
import nibabel
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class DataSet(Dataset):
    def __init__(self, root_path, dir):
        """
        Dataset class for loading medical imaging data
        :param root_path: Root directory containing data
        :param dir: Subdirectory containing images for a specific class
        """
        self.root_path = root_path
        self.dir = dir
        self.image_path = os.path.join(self.root_path, dir)
        self.images = os.listdir(self.image_path)  # List all files in directory

    def __getitem__(self, index):
        """Load and preprocess a single image and its label"""
        label = 0
        image_index = self.images[index]  # Get filename at index
        img_path = os.path.join(self.image_path, image_index)  # Full image path
        img = nibabel.load(img_path).get_fdata()  # Load NIfTI image data
        img = img.astype('float32')  # Convert to float32

        # Normalization options
        normalization = 'minmax'
        if normalization == 'minmax':
            # Min-max normalization
            img_max = img.max()
            img = img / img_max
        elif normalization == 'median':
            # Normalize by median (excluding zeros)
            img_fla = np.array(img).flatten()
            index = np.argwhere(img_fla == 0)
            img_median = np.median(np.delete(img_fla, index))
            img = img / img_median

        img = np.expand_dims(img, axis=0)  # Add channel dimension

        # Assign label based on directory name
        if self.dir == 'AD/':
            label = 1  # Alzheimer's Disease
        elif self.dir == 'CN/':
            label = 0  # Cognitively Normal
        elif self.dir == 'MCI/':
            label = 2  # Mild Cognitive Impairment

        # Clean up temporary variables
        if normalization == 'minmax':
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()  # Force garbage collection

        return img, label

    def __len__(self):
        """Return total number of images in dataset"""
        return len(self.images)


def load_data(args, root_path, path1, path2):
    """
    Create DataLoader from two datasets
    :param args: Configuration arguments
    :param root_path: Root directory containing data
    :param path1: Path to first class data (e.g., AD)
    :param path2: Path to second class data (e.g., CN)
    :return: DataLoader for combined dataset
    """
    # Create datasets for both classes
    train_AD = DataSet(root_path, path1)
    train_CN = DataSet(root_path, path2)

    # Combine datasets
    trainDataset = train_AD + train_CN

    # Create DataLoader
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

    # Clean up
    del trainDataset
    gc.collect()
    return train_loader

# Example usage:
# args = get_args()
# train_data, test_data = load_data(args)
# for step, (b_x, b_y) in enumerate(train_data):
#     if step > 1:
#         break
#
# print(b_x.shape)
# print(b_y.shape)
# print(b_x)
# print(b_y)