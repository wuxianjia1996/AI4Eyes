# eye_images
deep learning for eye images analysis

## Installation 

Pytorch 1.10 + CUDA10.2 

```python
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Cataract and Detach_retinal

### Data pre-processing

Split into train / val / test

```python
cd misc
python pre_proces.py --help
```

For cataract and detach_retinal, by running : 
```python
python pre_process.py --img-dir cataract_org --out-dir cataract_img
python pre_process.py --img-dir detach_retinal_org --out-dir detach_retinal_img
```

We should obtain : 

```python
['cataract', 'normal']
cls name cataract : 
    train images --> 1196; 
    val images --> 170;
    test images --> 343
cls name normal : 
    train images --> 140; 
    val images --> 20; 
    test images --> 41
    
['detach_retinal', 'normal']
cls name detach_retinal : 
    train images --> 244; 
    val images --> 35;
    test images --> 71
cls name normal : 
    train images --> 280; 
    val images --> 40; 
    test images --> 80
```


## Training

To see the command: 
```python
python train.py --help
```

Example of training command (training with imagenet pretrained weight and batch size 64): 
```python
python train.py --out-dir resnet18_inetpre_iter1k --cuda --batch-size 64 --inet-pretrain
```

After the training, to save the parameters of trained model:
```python
torch.save(net.state_dict(), 'net_cataract.pt')
torch.save(net.state_dict(), 'net_retinal.pt')
```


## Test

To test the trained model, load saved parameters:
```python
new_net = models.resnet18(pretrained= args.inet_pretrain)
new_net.fc = nn.Linear(512, args.nb_cls)
new_net.load_state_dict(torch.load('net_cataract.pt'))
```

## Class Activation Mapping (CAM)
Loading the trained model to visualize the discriminative image regions used by the CNN to identify image category.
