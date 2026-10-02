import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
import numpy as np
import random
from typing import List, Tuple

class ImageEditingPrimitives:
    """Implements the 8 ground-truth image editing primitives defined in the L2T paper."""
    
    @staticmethod
    def brightness_plus(img: torch.Tensor) -> torch.Tensor:
        # Increase brightness by a factor of 1.5
        return F.adjust_brightness(img, 1.5)

    @staticmethod
    def brightness_minus(img: torch.Tensor) -> torch.Tensor:
        # Decrease brightness by a factor of 0.5
        return F.adjust_brightness(img, 0.5)

    @staticmethod
    def hue_plus(img: torch.Tensor) -> torch.Tensor:
        # Rotate hue by +0.3 on the color wheel
        return F.adjust_hue(img, 0.3)

    @staticmethod
    def hue_minus(img: torch.Tensor) -> torch.Tensor:
        # Rotate hue by -0.3 on the color wheel
        return F.adjust_hue(img, -0.3)

    @staticmethod
    def horizontal_flip(img: torch.Tensor) -> torch.Tensor:
        # Flip image left-right
        return F.hflip(img)

    @staticmethod
    def vertical_flip(img: torch.Tensor) -> torch.Tensor:
        # Flip image top-bottom
        return F.vflip(img)

    @staticmethod
    def rotation(img: torch.Tensor) -> torch.Tensor:
        # Rotate image by 90 degrees clockwise (torchvision uses counter-clockwise, so -90)
        return F.rotate(img, -90)

    @staticmethod
    def masking(img: torch.Tensor) -> torch.Tensor:
        # Apply a gray (128/255) square mask to the top-left quadrant
        img_out = img.clone()
        _, h, w = img_out.shape
        # Top-left quadrant is 25% of the image (h//2, w//2)
        img_out[:, :h//2, :w//2] = 128.0 / 255.0
        return img_out


class OTIBDatasetGenerator:
    """Generates (x, y) observation pairs from CIFAR-10 based on latent programs."""
    
    def __init__(self, data_root: str = './data', diff_threshold: float = 0.05):
        self.diff_threshold = diff_threshold
        
        # Load raw CIFAR-10 (converted to float Tensors in [0, 1])
        self.cifar10 = CIFAR10(
            root=data_root, 
            train=True, 
            download=True, 
            transform=torchvision.transforms.ToTensor()
        )
        
        # Map primitive names to functions
        self.primitives_map = {
            'br_p': ImageEditingPrimitives.brightness_plus,
            'br_m': ImageEditingPrimitives.brightness_minus,
            'hue_p': ImageEditingPrimitives.hue_plus,
            'hue_m': ImageEditingPrimitives.hue_minus,
            'h_flip': ImageEditingPrimitives.horizontal_flip,
            'v_flip': ImageEditingPrimitives.vertical_flip,
            'rot': ImageEditingPrimitives.rotation,
            'mask': ImageEditingPrimitives.masking
        }
        self.primitive_names = list(self.primitives_map.keys())

    def apply_program(self, x: torch.Tensor, program: List[str]) -> torch.Tensor:
        """Executes a sequence of primitive operations on an image."""
        y = x.clone()
        for prim_name in program:
            y = self.primitives_map[prim_name](y)
        return y

    def is_valid_transformation(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Filters out transformations where the pixel-wise difference is too small."""
        mse_diff = torch.nn.functional.mse_loss(x, y).item()
        return mse_diff > self.diff_threshold

    def generate_pairs(self, num_samples: int, program_lengths: List[int]) -> List[dict]:
        """Generates a dataset of (source, target, latent_program) dicts."""
        dataset = []
        attempts = 0
        
        while len(dataset) < num_samples:
            # 1. Pick a random source image
            idx = random.randint(0, len(self.cifar10) - 1)
            x, _ = self.cifar10[idx]
            
            # 2. Sample a random program length and primitive sequence
            length = random.choice(program_lengths)
            program = random.choices(self.primitive_names, k=length)
            
            # 3. Execute the theory (apply the program to get target y)
            y = self.apply_program(x, program)
            
            # 4. Check semantic relevance threshold
            if self.is_valid_transformation(x, y):
                dataset.append({
                    'x': x,
                    'y': y,
                    'program': program
                })
            
            attempts += 1
            if len(dataset) % 1000 == 0 and len(dataset) > 0:
                print(f"Generated {len(dataset)}/{num_samples} pairs... (Attempts: {attempts})")
                
        return dataset

# --- Example Usage ---
if __name__ == "__main__":
    generator = OTIBDatasetGenerator(diff_threshold=0.01)
    
    # 1. Generate IID Training Data (Lengths 1 & 2)
    print("Generating Training Data...")
    train_data = generator.generate_pairs(num_samples=10000, program_lengths=[1, 2])
    
    # 2. Generate Length OOD Test Data (Lengths 3 & 4)
    print("Generating Length OOD Test Data...")
    length_ood_data = generator.generate_pairs(num_samples=2000, program_lengths=[3, 4])
    
    print("\nSample Training Pair:")
    sample = train_data[0]
    print(f"Source Image Shape: {sample['x'].shape}")
    print(f"Target Image Shape: {sample['y'].shape}")
    print(f"Latent Program Executed: {sample['program']}")
    
    # To save the dataset for PyTorch DataLoader usage:
    # torch.save(train_data, 'otib_train_cifar10.pt')
    # torch.save(length_ood_data, 'otib_length_ood_cifar10.pt')