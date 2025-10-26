import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import custom modules from local files
from Net.PatchFeatureExtractor import BaseNet
from Net.Patch_Blocking import Patches_CropSelected
from Net.attention import VisionTransformerBlock
from Net.crosstransformer import CrossTransformer_MOD_AVG


class NET_MRI_1(nn.Module):
    def __init__(self, class_num, input_shape, patch_size):
        """
        MRI classification network with left/right hemisphere processing
        :param class_num: Number of output classes
        :param input_shape: Input volume shape [depth, height, width]
        :param patch_size: Size of patches to extract from input volume
        """
        super().__init__()
        self.input_shape = input_shape

        # Calculate cropping parameters to make dimensions divisible by patch size
        self.input_cut = [(input_shape[i] % patch_size) for i in range(3)]
        self.input_cut1 = [self.input_cut[i] // 2 for i in range(3)]
        self.input_cut2 = [self.input_cut[i] - self.input_cut1[i] for i in range(3)]

        # Adjust cropping indices for each dimension
        for i in range(3):
            if self.input_cut2[i] == 0:
                self.input_cut2[i] = None  # No cropping needed
            else:
                self.input_cut2[i] = -self.input_cut2[i]  # Negative index for end cropping

        # Calculate shapes for left and right hemispheres after cropping
        input1_shape = [input_shape[i] - self.input_cut[i] for i in range(3)]
        input2_shape = [input_shape[i] - self.input_cut[i] for i in range(3)]
        input1_shape[0] = input1_shape[0] // 2  # Left hemisphere depth
        input2_shape[0] = input2_shape[0] // 2  # Right hemisphere depth

        # Patch extraction modules for left and right hemispheres
        self.input1 = Patches_CropSelected(input_shape=input1_shape, patch_size=patch_size)
        self.input2 = Patches_CropSelected(input_shape=input2_shape, patch_size=patch_size)

        # Feature extractors for patches
        self.patchFeatureExtractorLeft = BaseNet([32, 64, 128, 128])
        self.patchFeatureExtractorRight = BaseNet([32, 64, 128, 128])

        # Calculate number of patches per hemisphere
        self.LeftPatchNumber = torch.prod(self.input1.patches_shape)
        self.RightPatchNumber = torch.prod(self.input2.patches_shape)

        # Attention modules for each hemisphere and cross-modal attention
        self.attentionLeft = VisionTransformerBlock(128, 8)
        self.attentionRight = VisionTransformerBlock(128, 8)
        self.attentionGlobal = CrossTransformer_MOD_AVG(128, 4, 8, 64, 512, 0.1)

        # Classification heads for each hemisphere and combined features
        self.classifierLeft = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor([self.LeftPatchNumber, 128])), 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1),
        )
        self.classifierRight = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor([self.LeftPatchNumber, 128])), 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(True),
            nn.Linear(32, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """Forward pass through the network"""
        b = x.shape[0]  # Batch size

        # Crop input to make dimensions divisible by patch size
        x = x[:, :, self.input_cut1[0]:self.input_cut2[0],
            self.input_cut1[1]:self.input_cut2[1],
            self.input_cut1[2]:self.input_cut2[2]]

        # Split into left and right hemispheres
        x1 = x[:, :, :(x.shape[2] // 2), :, :]  # Left hemisphere
        x2 = x[:, :, (x.shape[2] // 2):, :, :]  # Right hemisphere

        # Extract patches from each hemisphere
        x1 = self.input1(x1)
        x2 = self.input2(x2)

        # Rearrange patches for batch processing
        x1 = rearrange(x1, 'b num (c h) w d -> (b num) c h w d', c=1)
        x2 = rearrange(x2, 'b num (c h) w d -> (b num) c h w d', c=1)

        # Extract features from patches
        x1, x1_score = self.patchFeatureExtractorLeft(x1)
        x2, x2_score = self.patchFeatureExtractorRight(x2)

        # Global average pooling to reduce spatial dimensions
        x1 = F.adaptive_avg_pool3d(x1, (1, 1, 1))
        x2 = F.adaptive_avg_pool3d(x2, (1, 1, 1))

        # Flatten features for later use
        left_features_flat = x1.view(-1, 128)
        right_features_flat = x2.view(-1, 128)

        # Rearrange for attention processing
        x1 = rearrange(x1, '(b num) c h w d -> b num (c h w d)', b=b)
        x2 = rearrange(x2, '(b num) c h w d -> b num (c h w d)', b=b)

        # Apply attention mechanisms
        x1 = self.attentionLeft(x1)
        x2 = self.attentionRight(x2)
        cls, _, _ = self.attentionGlobal(x1, x2)  # Cross-modal attention

        # Flatten for classification
        x1 = rearrange(x1, 'b num l -> b (num l)', b=b)
        x2 = rearrange(x2, 'b num l -> b (num l)', b=b)

        # Classification outputs
        x1 = self.classifierLeft(x1)
        x2 = self.classifierRight(x2)
        cls = self.classifier(cls)

        return left_features_flat, right_features_flat, x1, x2, cls


if __name__ == '__main__':
    # Test the network with sample input
    x = torch.randn(3, 1, 121, 145, 121)
    model = NET_MRI_1(2, input_shape=[121, 145, 121], patch_size=30)
    left, right, y1, y2, output = model(x)
    print(output)