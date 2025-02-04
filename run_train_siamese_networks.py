import hydra
import lightning as L
from torch.utils.data import DataLoader

from src.dataset.siamese_dataset import SiameseDataset
from src.dataset.mnist_example_dataset import APP_MATCHER
from src.models.siamese_nets import LitSiameseNets


@hydra.main(version_base=None, config_path="configs", config_name="siamese_train_config")
def main(config):

    # transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train_dataset = SiameseDataset()
    # test_dataset = SiameseDataset()

    train_dataset = APP_MATCHER('../data', train=True, download=True)
    test_dataset = APP_MATCHER('../data', train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = LitSiameseNets(config)
    trainer = L.Trainer(**config.trainer.params)

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()

