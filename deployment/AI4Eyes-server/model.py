import torchvision.models as models
import torch
import warnings
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os

warnings.filterwarnings("ignore", category=UserWarning)


class EyeModel:
    def __init__(self, nb_cls: int = 2):
        self.nb_cls = nb_cls
        self.inet_pretrain = True
        self.model2 = models.resnet18(pretrained=self.inet_pretrain)
        self.model2.fc = torch.nn.Linear(512, self.nb_cls)
        self.model2.load_state_dict(
            torch.load(r'./ckpt/2cls_net_cataract.pt', map_location=torch.device('cpu')))
        self.model2.eval()

        self.model3 = models.resnet18(pretrained=self.inet_pretrain)
        self.model3.fc = torch.nn.Linear(512, 3)
        self.model3.load_state_dict(
            torch.load(r"./ckpt/3cls_net_cataract.pth", map_location=torch.device('cpu')))
        self.model3.eval()

        self.model4 = models.resnet18(pretrained=self.inet_pretrain)
        self.model4.fc = torch.nn.Linear(512, 4)
        self.model4.load_state_dict(
            torch.load(r"./ckpt/4cls_net_cataract.pth", map_location=torch.device('cpu')))
        self.model4.eval()

        print('models loaded')

    def predict2(self, img_tensor):
        with torch.no_grad():
            outputs = self.model2(img_tensor.unsqueeze(0))
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            return probs, preds

    def predict3(self, img_tensor):
        with torch.no_grad():
            outputs = self.model3(img_tensor.unsqueeze(0))
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            return probs, preds

    def predict4(self, img_tensor):
        with torch.no_grad():
            outputs = self.model4(img_tensor.unsqueeze(0))
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            return probs, preds
