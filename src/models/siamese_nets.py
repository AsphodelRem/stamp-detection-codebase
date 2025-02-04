import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import torchvision
import lightning as L

from src.loss.contrastive_loss import ContrastiveLoss


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)

        output = self.fc(output)
        output = self.sigmoid(output)
        
        return output


class LitSiameseNets(L.LightningModule):
    def __init__(self, config):
        super(LitSiameseNets, self).__init__()
        self.model = SiameseNetwork()
        self.criterion = ContrastiveLoss()
        self.config = config
    
    def forward(self, x1, x2):
        return self.model(x1, x2)
    
    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        outputs = self(img1, img2)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return AdamW(
            params=self.model.parameters(), 
            **self.config.trainer.optimizer
        )


