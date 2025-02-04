import hydra
import comet_ml
import lightning as L
from torch.utils.data import DataLoader
from  pytorch_lightning.loggers import CometLogger

from src.dataset.siamese_dataset import SiameseDataset
from src.dataset.mnist_example_dataset import APP_MATCHER
from src.models.siamese_nets import LitSiameseNets



@hydra.main(version_base=None, config_path="configs", config_name="siamese_train_config")
def main(config):
    comet_logger = CometLogger(project_name="comet-examples-lightning")

    # transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train_dataset = SiameseDataset()
    # test_dataset = SiameseDataset()

    train_dataset = APP_MATCHER('../data', train=True, download=True)
    test_dataset = APP_MATCHER('../data', train=False)

    train_dataloader = DataLoader(train_dataset, **config.dataloader)
    test_dataloader = DataLoader(test_dataset, **config.dataloader)

    model = LitSiameseNets(config)
    trainer = L.Trainer(logger=comet_logger, **config.trainer.params)

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()

