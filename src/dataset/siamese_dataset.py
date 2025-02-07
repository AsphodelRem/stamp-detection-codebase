from pathlib import Path
import random
import itertools

import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset
from PIL import Image


def find_anchor(folder):
    for img_path in folder.glob('*'):
        if img_path.stem in ["anchor", "ancor"]:
            return img_path
    return None


def find_images(folder):
    for img_path in folder.glob('*'):
        if (img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']) and not (img_path.stem in ["anchor", "ancor"]):
            return img_path
    return None


def group_classes(root_dir):

    return {
        f"{class_folder.name}/{subclass_folder.name}": {
            "images": [img_path for img_path in subclass_folder.glob('*')
                       if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']],
            "anchor": find_anchor(subclass_folder)
        }
        for class_folder in root_dir.iterdir() if class_folder.is_dir()
        for subclass_folder in class_folder.iterdir() if subclass_folder.is_dir()
    }


def generate_image_pairs(class_to_images: dict[str, list[str]], num_different: int, seed: int = None) -> list[tuple[str, str, int]]:
    """
    Генерирует пары изображений (img1, img2, label), где label=1 для одинаковых классов и -1 для разных.

    Args:
        class_to_images (dict): Словарь {класс: [список_изображений]}
        num_different (int): Количество пар с разными классами (label=0)
        seed (int, optional): Фиксация случайности

    Returns:
        list[tuple[str, str, int]]: Список пар изображений с метками (img1, img2, label)
    """
    if seed:
        random.seed(seed)

    pairs = set()
    classes = list(class_to_images.keys())

    # Пары с одинаковыми классами (label = 1)
    same_pairs = []
    for class_name, data in class_to_images.items():
        for img in data["images"]:
            same_pairs.append((data["anchor"], img, 1))

    # Пары с разными классами (label = -1)
    different_pairs = []
    for class1, class2 in itertools.combinations(classes, 2):
        images1 = class_to_images[class1]["images"]
        images2 = class_to_images[class2]["images"]

        for img1, img2 in itertools.product(images1, images2):
            different_pairs.append((img1, img2, -1))

    # Ограничиваем количество пар с разными классами
    different_pairs = random.sample(
        different_pairs, min(num_different, len(different_pairs)))

    # Собираем итоговый список пар
    pairs.update(same_pairs)
    pairs.update(different_pairs)

    return list(pairs)


def apply_augmentations(paths, augmentations):
    """Применяет аугментации к изображениям."""
    augmentation_pipeline = A.Compose(augmentations)
    images = [np.array(Image.open(img).convert("RGB")) for img in paths]
    augmented = [augmentation_pipeline(image=img) for img in images]
    # HWC -> CHW
    return [torch.tensor(img["image"]).permute(2, 0, 1) for img in augmented]


class SiameseDataset(Dataset):
    augmentations = [
        A.Resize(128, 128),  # Изменение размера
        A.Blur(blur_limit=5, p=0.3),  # Размытие
        A.AdditiveNoise(noise_type='gaussian', p=0.3),  # Гауссовский шум
        A.Rotate(limit=45, p=0.3),  # Повороты на ±45 градусов
        # Изменение яркости и контраста
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1),
                 rotate=(-20, 20), p=0.3),  # Вместо ShiftScaleRotate
        A.HorizontalFlip(p=0.3),  # Отражение по горизонтали
        A.ToFloat(),  # Приведение к float32 [0,1] (подготовка для тензоров)
    ]
    
    def __init__(self, root_dir, num_different):
        self.root_dir = Path(root_dir)
        grouped = group_classes(self.root_dir)
        self.pairs = generate_image_pairs(
            grouped,
            num_different=num_different
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path1, path2, target = self.pairs[index]
        img1, img2 = apply_augmentations(
            [path1, path2],
            self.augmentations,
        )
        return img1, img2, target
