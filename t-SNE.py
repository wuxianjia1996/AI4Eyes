import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


model = models.resnet18(pretrained= True)
model.fc = nn.Linear(512, 4)
model.load_state_dict(torch.load('checkpoint_path'))
model.eval()
device = torch.device('cuda:0')
model = model.to(device)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,])

dataset = datasets.ImageFolder('test_dir path', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

features = []
lables = []

with torch.no_grad():
    for images, target in dataloader:
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        output = output.squeeze()
        features.append(output)
        lables.append(target)

features = torch.cat(features).detach().cpu().view(len(dataset), -1)
lables = torch.cat(lables).detach().cpu().numpy()

tsne = TSNE(n_components=4, random_state=0)
features_tsne = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
for i in range(len(dataset.classes)):
    plt.scatter(features_tsne[lables == i, 0], features_tsne[lables == i, 1], label=dataset.classes[i])

plt.legend()
plt.savefig('output/tsne_4cls_test.png')
plt.show()