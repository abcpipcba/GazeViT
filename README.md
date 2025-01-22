# GazeViT
This repository provides the code for "[GazeViT: a gaze-guided hybrid-attention Vision Transformer for cross-view matching of street-to-aerial images]

<img width=1549 height=1974 src="data/Overview.png"/>

## Dataset
Please prepare [CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/) or [CVACT](https://github.com/Liumouliu/OriCNN). You may need to modify specific path in dataloader.

## Requirement
	- Python >= 3.6, numpy, matplotlib, pillow, ptflops, timm
    - PyTorch >= 1.8.1, torchvision >= 0.11.1
	
## Training and Evaluation
Simply run the scripts like:
    run train.py

Change the "--crop" to be False if you want to train the  stage 1 model. 

We follow timm, ViT and [Deit](https://github.com/facebookresearch/deit) for pytorch implementation of vision transformer. We use the pytorch implementation of [ASAM](https://github.com/davda54/sam).

Let me know if there is anything wrong with the link.
    
## Reference
    - http://mvrl.cs.uky.edu/datasets/cvusa/
    - https://github.com/Jeff-Zilence/TransGeo2022
    - https://github.com/Liumouliu/OriCNN
    - https://github.com/facebookresearch/deit
    - https://github.com/davda54/sam
    - https://github.com/david-husx/crossview_localisation.git

