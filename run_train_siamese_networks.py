import comet_ml
import hydra
import lightning as L
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.dataset.siamese_dataset import SiameseDataset
from src.models.siamese_nets import LitSiameseNets


@hydra.main(version_base=None, config_path="configs", config_name="siamese_train_config")
def main(config):
    comet_logger = CometLogger(project_name="comet-examples-lightning")

    dataset = SiameseDataset(
        root_dir="./data/stamps_2",
        num_different=1200
    )
    print(len(dataset))
    train_size = int(config.trainer.dataset.train_size * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, **config.dataloader)
    test_dataloader = DataLoader(test_dataset, **config.dataloader)

    model = LitSiameseNets(config)
    trainer = L.Trainer(
        logger=comet_logger,
        **config.trainer.params
    )

    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.validate(model, test_dataloader)

    import torch
    PATH = "model_state_dict.pth"

    # Save the state dictionary
    torch.save(model.model.resnet.state_dict(), PATH)


if __name__ == "__main__":
    main()
