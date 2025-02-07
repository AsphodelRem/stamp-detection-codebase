from PIL import Image
from torchvision import transforms
from pathlib import Path
import shutil

from src.dataset.siamese_dataset import SiameseDataset


dataset = SiameseDataset(
    root_dir="./data/stamps_2",
    num_different=1200,
)
dir = Path("./data/test-images")
if dir.exists() and dir.is_dir():
    shutil.rmtree(dir)
dir.mkdir(parents=True, exist_ok=True)

print(len(dataset))


def tensor_to_pil(tensor):
    """Преобразует тензор (C, H, W) в PIL-изображение"""
    tensor = tensor.clone().detach()  # Создаём копию, чтобы не изменять исходный тензор
    tensor = tensor.permute(1, 2, 0)  # CHW → HWC
    tensor = tensor.numpy()  # Преобразуем в numpy
    # Масштабируем пиксели обратно в [0, 255]
    tensor = (tensor * 255).astype("uint8")

    return Image.fromarray(tensor)


for i in range(len(dataset)):
    img1, img2, label = dataset.__getitem__(i)
    img1, img2 = tensor_to_pil(img1), tensor_to_pil(img2)
    new_img = Image.new('RGB', (img1.width + img2.width,
                        max(img1.height, img2.height)))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    
    if label == 1:
        new_img.save(dir / f"simillar-{i}.png")
    else:
        new_img.save(dir / f"different-{i}.png")

        
