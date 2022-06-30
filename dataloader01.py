from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def TrainLoader(batch_size, img_dir, train_transform):

    dataloader = DataLoader(ImageFolder(img_dir, train_transform), batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader

def ValLoader(batch_size, img_dir, val_transform):

    dataloader = DataLoader(ImageFolder(img_dir, val_transform), batch_size=batch_size, shuffle=False)

    return dataloader

def TestLoader(batch_size, img_dir, test_transform):

    dataloader = DataLoader(ImageFolder(img_dir, test_transform), batch_size=batch_size, shuffle=False)

    return dataloader