import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RiceDataset(Dataset):
    """
    Dataset loader for satellite cloud removal (RICE1, RICE2, or Combined).
    Supports data augmentations (flips, rotations, contrast) and train/val splits.
    """

    def __init__(
        self,
        dataset_paths,
        split="all",
        val_ratio=0.15,
        seed=42,
        is_train=False,
        img_size=None
    ):
        """
        Args:
            dataset_paths: str or list of paths (e.g. [path_rice1, path_rice2])
            split: 'train', 'val', or 'all'
            val_ratio: fraction of data to use for validation
            seed: random seed for deterministic split
            is_train: if True, applies random augmentations
            img_size: optional (H, W) tuple to resize images
        """
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        self.samples = []
        self.is_train = is_train
        self.img_size = img_size

        for base_path in dataset_paths:
            cloud_dir = os.path.join(base_path, "cloud")
            label_dir = os.path.join(base_path, "label")

            if not os.path.exists(cloud_dir) or not os.path.exists(label_dir):
                continue

            cloud_files = set(os.listdir(cloud_dir))
            label_files = set(os.listdir(label_dir))
            common = sorted(list(cloud_files.intersection(label_files)))

            for fname in common:
                # verify image extensions
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                    self.samples.append((
                        os.path.join(cloud_dir, fname),
                        os.path.join(label_dir, fname)
                    ))

        # Deterministic split
        if split in ("train", "val") and len(self.samples) > 0:
            rng = random.Random(seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            val_count = max(1, int(len(self.samples) * val_ratio))

            if split == "val":
                selected_indices = set(indices[:val_count])
            else:
                selected_indices = set(indices[val_count:])

            self.samples = [self.samples[i] for i in sorted(selected_indices)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cloud_path, label_path = self.samples[index]

        cloud = cv2.imread(cloud_path)
        label = cv2.imread(label_path)

        if cloud is None:
            raise FileNotFoundError(f"Could not read {cloud_path}")
        if label is None:
            raise FileNotFoundError(f"Could not read {label_path}")

        cloud = cv2.cvtColor(cloud, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.img_size is not None:
            cloud = cv2.resize(cloud, self.img_size, interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, self.img_size, interpolation=cv2.INTER_AREA)

        # Data Augmentations during training
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                cloud = np.fliplr(cloud).copy()
                label = np.fliplr(label).copy()

            # Random vertical flip
            if random.random() > 0.5:
                cloud = np.flipud(cloud).copy()
                label = np.flipud(label).copy()

            # Random 90 deg rotation
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                cloud = np.rot90(cloud, k).copy()
                label = np.rot90(label, k).copy()

        cloud = cloud.astype("float32") / 255.0
        label = label.astype("float32") / 255.0

        cloud_tensor = torch.from_numpy(cloud).permute(2, 0, 1)
        label_tensor = torch.from_numpy(label).permute(2, 0, 1)

        return cloud_tensor, label_tensor