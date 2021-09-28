import torch, torchvision
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
import PIL


model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(in_features=num_features, out_features=10)

        
model.load_state_dict(torch.load("static/resnet18.pth"))
model.eval()

img_transforms = transforms.Compose([transforms.Resize((32,32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                           [0.2023, 0.1994, 0.2010])])

def predict(full_path):
    data = PIL.Image.open(full_path)

    data = img_transforms(data)
    data = torch.unsqueeze(data, 0)

    with torch.no_grad():
        predicted = model(data)
        predicted = F.softmax(predicted, dim=1)
        return predicted