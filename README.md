# U2PL_YX
the reimplementation of U2PL model to test the M&amp;MS and SCGM dataset


# datasets
- Use [M&Ms](https://www.ub.edu/mnms/) and [SCGM](http://niftyweb.cs.ucl.ac.uk/challenge/index.php) dataset
- Use the same preprocess method as DGNet (code can be found [here](https://github.com/xxxliu95/DGNet/tree/main/preprocess))

# pretrained model
This model use the Deeplab-v3 model to complete the segmentation task. ResNet-101 pretrained weights (pretrained on ImageNet) is used in this model. The model can be downloaded [here](https://drive.google.com/file/d/1LqddoCjT0aVerRc93b4JrjvPiYhEFhRz/view?usp=sharing) and put this file under the './models/pretrained' folder

# How to run
If you want to train model on M&Ms dataset, you can run `python train_mnms.py`. Some parameters can be changed using args

# Main Results
The model cannot work well on SCGM dataset now and I'm still in experimenting. The following are some results on M&Ms dataset.
![Result1](https://user-images.githubusercontent.com/23032654/174572953-32ead5c6-131d-4bd7-a129-b2f998112db0.png)
![Result2](https://user-images.githubusercontent.com/23032654/174573005-a0be6247-39d9-4c25-9e0f-45ed216a1c5a.png)
