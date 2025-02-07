import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss
from torch.optim import AdamW
import numpy as np

import torchvision
import lightning as L

from src.loss.contrastive_loss import ContrastiveLoss


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')

        self.fc_in_features = self.resnet.fc.in_features


        # Список слоев модели
        layers = list(self.resnet.children())

        # Количество слоев, которые нужно заморозить
        N = 30

        # Замораживаем первые N слоев
        for layer in layers[:N]:
            for param in layer.parameters():
                param.requires_grad = False

        # Оставляем остальные слои обучаемыми
        for layer in layers[N:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # output = torch.cat((output1, output2), 1)

        # output = self.fc(output)
        # output = self.sigmoid(output)

        return (output1, output2)


class LitSiameseNets(L.LightningModule):
    def __init__(self, config):
        super(LitSiameseNets, self).__init__()
        self.model = SiameseNetwork()
        # self.criterion = ContrastiveLoss()
        self.criterion = CosineEmbeddingLoss(margin=0.5)
        self.config = config

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def step(self, batch, batch_idx, stage: str):
        img1, img2, labels = batch
        embedding_input, anchor_embedding = self(img1, img2)
        loss = self.criterion(embedding_input, anchor_embedding, labels)
        self.log(f"{stage}/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        """Один шаг валидации."""
        img1, img2, labels = batch
        # preds = self(img1, img2)
        loss = self.step(batch, batch_idx, 'val')

        # mae = self.compute_mae(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_mae", mae, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(
            params=self.model.parameters(),
            **self.config.trainer.optimizer
        )

    def compute_mae(self, preds, labels):
        """Вычисление F1 на топ-N."""
        print("Preds", preds)
        return np.mean(np.abs(labels.cpu().numpy() - preds.cpu().numpy()))
