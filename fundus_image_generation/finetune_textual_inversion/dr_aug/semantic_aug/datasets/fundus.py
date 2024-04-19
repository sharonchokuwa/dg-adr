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


DEFAULT_DATA_DIR =  r"../finetune_dreambooth/eyepacs/instance_images"

class FundusDataset(FewShotDataset):

    num_classes: int = 5
    class_names = ["no diabetic retinopathy", 
                   "mild diabetic retinopathy",
                   "moderate diabetic retinopathy",
                   "severe diabetic retinopathy",
                   "proliferative diabetic retinopathy",]

    def __init__(self, *args, data_dir: str = DEFAULT_DATA_DIR, 
                 split: str = "train", seed: int = 0, 
                 examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (512, 512), **kwargs):

        super(FundusDataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)


        image_formats = ('jpg', 'jpeg', 'png')
        no_dr = []
        mild_dr = []
        moderate_dr = []
        severe_dr = []
        proliferative_dr = []

        for file_type in image_formats:
            no_dr.extend(glob.glob(os.path.join(data_dir, f"0/*.{file_type}")))
            mild_dr.extend(glob.glob(os.path.join(data_dir, f"1/*.{file_type}")))
            moderate_dr.extend(glob.glob(os.path.join(data_dir, f"2/*.{file_type}")))
            severe_dr.extend(glob.glob(os.path.join(data_dir, f"3/*.{file_type}")))
            proliferative_dr.extend(glob.glob(os.path.join(data_dir, f"4/*.{file_type}")))

        rng = np.random.default_rng(seed)

        no_dr_ids = rng.permutation(len(no_dr))
        mild_dr_ids = rng.permutation(len(mild_dr))
        moderate_dr_ids = rng.permutation(len(moderate_dr))
        severe_dr_ids = rng.permutation(len(severe_dr))
        proliferative_dr_ids = rng.permutation(len(proliferative_dr))

        no_dr_ids_train, no_dr_ids_val = np.array_split(no_dr_ids, 2)
        mild_dr_ids_train, mild_dr_ids_val = np.array_split(mild_dr_ids, 2)
        moderate_dr_ids_train, moderate_dr_ids_val = np.array_split(moderate_dr_ids, 2)
        severe_dr_ids_train, severe_dr_ids_val = np.array_split(severe_dr_ids, 2)
        proliferative_dr_ids_train, proliferative_dr_ids_val = np.array_split(proliferative_dr_ids, 2)

        no_dr_ids = {"train": no_dr_ids_train, "val": no_dr_ids_val}[split]
        mild_dr_ids = {"train": mild_dr_ids_train, "val": mild_dr_ids_val}[split]
        moderate_dr_ids = {"train": moderate_dr_ids_train, "val": moderate_dr_ids_val}[split]
        severe_dr_ids = {"train": severe_dr_ids_train, "val": severe_dr_ids_val}[split]
        proliferative_dr_ids = {"train": proliferative_dr_ids_train, "val": proliferative_dr_ids_val}[split]

        if examples_per_class is not None:
            no_dr_ids = no_dr_ids[:examples_per_class]
            mild_dr_ids = mild_dr_ids[:examples_per_class]
            moderate_dr_ids = moderate_dr_ids[:examples_per_class]
            severe_dr_ids = severe_dr_ids[:examples_per_class]
            proliferative_dr_ids = proliferative_dr_ids[:examples_per_class]

        self.no_dr = [no_dr[i] for i in no_dr_ids]
        self.mild_dr = [mild_dr[i] for i in mild_dr_ids]
        self.moderate_dr = [moderate_dr[i] for i in moderate_dr_ids]
        self.severe_dr = [severe_dr[i] for i in severe_dr_ids]
        self.proliferative_dr = [proliferative_dr[i] for i in proliferative_dr_ids]

        self.all_images = self.no_dr + self.mild_dr + self.moderate_dr + self.severe_dr + self.proliferative_dr
        self.all_labels = [0] * len(self.no_dr) + [1] * len(self.mild_dr) + [2] * len(self.moderate_dr) + [3] * len(self.severe_dr) + [4] * len(self.proliferative_dr)

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