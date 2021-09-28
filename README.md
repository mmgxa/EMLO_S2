# EMLO Session 2

The objective of this class was to train a resent model on CIFAR10 and deploy it to Heroku (with the ability to randomly upload an image for classificaiton)


# Dataset

The CIFAR10 dataset from torchvision library was used to train the model

```python
train_dataset = torchvision.datasets.CIFAR10(
                    root='.',
                    train=True,
                    transform=transform_train, 
                    download=True
                    )
```

whereas transformations were used as data augmentation to improve the model's performance

```python
transform_train = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], 
                                    [0.2023, 0.1994, 0.2010]),
                                ])
```

The validation and testing dataset didn't involve any transformations

```python
transform_val = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], 
                                    [0.2023, 0.1994, 0.2010]),
                                ])
```


To find which label has been assigned to each class, we can run the following code

```python
print(train_dataset.class_to_idx)
```

The training loop has been written using PyTorch Lightning.

# Model
The model used was a pretrained Resnet18. All layers (except) for the batch normalization layers were 'frozen' (i.e. made untrainable) since the Resnet18 has been trained on ImageNet.

However, it has 1000 output classes. We changed this last layer to output for 10 classes in our case.


```python
from torchvision import models

resnet18 = models.resnet18(pretrained=True)


for name, param in resnet18.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(in_features=num_features, out_features=10)
```

# Training

The model was trained for 50 epochs. The training was done on Colab and the corresponding model was downloaded and used for testing.

# Styling

A minimalistic makeover was done using CSS/Bootstrap to improve the UI

```html
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
```


# Steps for Deployment

After entering the virtual environment created for the assignment and installing the required dependencies, the following commands were executed in the shell.

```
heroku login -i
heroku create ___
heroku local
pip freeze > requirements.txt
- remove pkg_resources==0.0.0 line (if present)
- add +cpu next to torch and torchvision
git init
heroku git:remote -a ___ (same as in line 2)
git add .
git config user.name "____"
git config user.email "______"
git commit -m "Final App"
git push heroku master
git remote set-url origin git@github.com:mmgxa/EMLO_S2.git
git push origin master

```
