# eye_images
deep learning for eye images analysis

## Installation 

Pytorch 1.10 + CUDA10.2 

```python
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Cataract

### Data pre-processing

Split into train / val / test

```python
cd misc
python pre_proces.py --help
```

For cataract, by running : 
```python
python pre_process.py --img-dir cataract_org --out-dir cataract_img
```

We should obtain : 

```python
['cataract', 'normal']
cls name cataract : 
    train images --> 140; 
    val images --> 20;
    test images --> 40
cls name normal : 
    train images --> 141; 
    val images --> 20; 
    test images --> 41
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
