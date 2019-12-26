# Variational Autoencoder with Implicit Optimal Priors  
This is a pytorch implementation of the following paper [[AAAI]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4439) [[arXiv]](https://arxiv.org/abs/1809.05284):  
```
@inproceedings{takahashi2019variational,
  title={Variational Autoencoder with Implicit Optimal Priors},
  author={Takahashi, Hiroshi and Iwata, Tomoharu and Yamanaka, Yuki and Yamada, Masanori and Yagi, Satoshi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={5066--5073},
  year={2019}
}
```
Please read license.txt before reading or using the files.  

## Prerequisites  
Please install `python>=3.6`, `torch`, `torchvision`, `numpy` and `scipy`.  

## Usage  
```
usage: main.py [-h] [--dataset DATASET] [--prior PRIOR]
               [--learning_rate LEARNING_RATE] [--seed SEED]
```
- You can choose the `dataset` from following four image datasets: `MNIST`, `OMNIGLOT`, `Histopathology` and `FreyFaces`.  
- You can choose the `prior` of the VAE from `normal` (standard Gaussian prior) or `iop` (implicit optimal prior).  
- You can also change the random `seed` of the training and `learning_rate` of the optimizer (Adam).  


## Example  
MNIST with standard Gaussian prior:  
```
python main.py --dataset MNIST --prior normal
```
MNIST with implicit optimal prior:  
```
python main.py --dataset MNIST --prior iop
```

## Output  
- After the training, the mean of log-likelihood for test dataset will be displayed.  
- The detailed information of the training and test will be saved in `npy` directory.  
