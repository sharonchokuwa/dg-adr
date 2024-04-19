from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple
from torch.utils.data import Dataset
from PIL import Image

import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch


DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), 'fundus_data/')

class FundusDRDataset_Generic(FewShotDataset):

    num_classes: int = 5
    class_names = ["no diabetic retinopathy", 
                   "mild diabetic retinopathy",
                   "moderate diabetic retinopathy",
                   "severe diabetic retinopathy",
                   "proliferative diabetic retinopathy",]

    def __init__(self, *args, data_dir: str = None, 
                 dr_grade: int = 0,
                 split: str = "train", seed: int = 0, 
                 grade_name: str = None,
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (512, 512), **kwargs):

        super(FundusDRDataset_Generic, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug,  **kwargs)

        image_formats = ('jpg', 'jpeg', 'png')
        dr = []
        
        for file_type in image_formats:
            dr.extend(glob.glob(os.path.join(data_dir, f"{dr_grade}/*.{file_type}")))
            
        rng = np.random.default_rng(seed)
        dr_ids = rng.permutation(len(dr))
        dr_ids_train = dr_ids
        dr_ids = {"train": dr_ids_train}[split]

        if examples_per_class is not None:
            dr_ids = dr_ids[:examples_per_class]
           
        self.dr = [dr[i] for i in dr_ids]
       
        self.all_images = self.dr 
        self.all_labels = [dr_grade] * len(self.dr)
        self.grade_name = grade_name

        if use_randaugment: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        else: train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:  
        return Image.open(self.all_images[idx])

    def get_label_by_idx(self, idx: int) -> torch.Tensor: 
        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Any:
        return dict(name=self.class_names[self.all_labels[idx]])

    def get_image_path_by_idx(self, idx):
        return self.all_images[idx]