# Imports


import time
import shutil

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


import tableprint as tp
import torchmetrics


try:
    shutil.rmtree('csv_logs')
except:
    pass



# Dataset and DataLoader


transform_train = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                ])

transform_val = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                ])

train_dataset = torchvision.datasets.CIFAR10(
                    root='.',
                    train=True,
                    transform=transform_train, 
                    download=True
                    )


val_dataset = torchvision.datasets.CIFAR10(
                    root='.',
                    train=False,
                    transform=transform_val, 
                    download=True
                    )


num_workers = 2
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,  num_workers=num_workers)


# Model
resnet18 = models.resnet18(pretrained=True)


for name, param in resnet18.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(in_features=num_features, out_features=10)


class Model(pl.LightningModule):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.avg_train_loss = 0.
        self.avg_valid_loss = 0.
        self.table_context = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.start_time = 0
        self.end_time = 0
        self.epoch_mins = 0
        self.epoch_secs = 0
        self.table_context = None
        self.train_accm = torchmetrics.Accuracy()
        self.valid_accm = torchmetrics.Accuracy()
        self.train_acc = 0.
        self.valid_acc = 0.
        self.c0 = 0.
        self.c1 = 0.
        

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return optim


    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        _, predictions = torch.max(output, 1)
        acc_train = self.train_accm(predictions, target)
        loss = self.loss_fn(output, target)
        return {"loss": loss, "p": predictions, "y": target}
    
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        _, predictions = torch.max(output, 1)
        acc_train = self.valid_accm(predictions, target)
        loss_valid = self.loss_fn(output, target)
        return {"loss": loss_valid, "p": predictions, "y": target}


    def on_train_epoch_start(self) :
        self.start_time = time.time()


    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking:
          return
        
        self.avg_valid_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.valid_acc = (self.valid_accm.compute() * 100).item()
        self.valid_accm.reset()
          

    def training_epoch_end(self, outputs):
        self.avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.train_acc = (self.train_accm.compute() * 100).item()
        self.train_accm.reset()

    def on_train_epoch_end(self):
        self.end_time = time.time()
        self.epoch_mins, self.epoch_secs = self.epoch_time(self.start_time, self.end_time)
        time_int = f'{self.epoch_mins}m {self.epoch_secs}s'
    
        metrics = {'epoch': self.current_epoch+1, 'Train Acc': self.train_acc, 'Train Loss': self.avg_train_loss,  'Valid Acc': self.valid_acc, 'Valid Loss': self.avg_valid_loss}
        if self.table_context is None:
          self.table_context = tp.TableContext(headers=['epoch', 'Train Acc', 'Train Loss', 'Valid Acc', 'Valid Loss', 'Time'])
          self.table_context.__enter__()
        self.table_context([self.current_epoch+1, self.train_acc, self.avg_train_loss, self.valid_acc, self.avg_valid_loss, time_int])
        self.logger.log_metrics(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
          self.table_context.__exit__()

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    


model = Model(resnet18)



csvlogger = CSVLogger('csv_logs', name='EMLO_S2', version=0)
trainer = pl.Trainer(max_epochs=50, num_sanity_val_steps=0, logger=csvlogger, gpus=1, log_every_n_steps=1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

torch.save(resnet18.state_dict(), "resnet18.pth")

