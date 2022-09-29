# Repro-GANs

Reproduce various GANs with PyTorch. 



## Progress

- [x] GAN (vanilla GAN)
- [x] CGAN
- [x] DCGAN
- [x] WGAN
- [x] WGAN-GP
- [x] SNGAN
- [x] SNGAN-projection
- [x] ACGAN
- [x] SAGAN
- [x] LSGAN
- [x] VEEGAN



## Training

Run command:

```shell
python train.py \
    --config_path CONFIG_PATH \
    --model {gan,dcgan,cgan,acgan,wgan,wgan-gp,sngan,sngan-projection,lsgan,sagan,veegan}
```

Some examples of configuration files are under `./configs/`, you can directly use them to train the models, or modify them as you wish.



## Generation

Run command:

```shell
python generate.py \
    --model {gan,dcgan,cgan,acgan,wgan,wgan-gp,sngan,sngan-projection,lsgan,sagan,veegan} \
    --model_path MODEL_PATH \
    --mode {random,walk} \
    --save_path SAVE_PATH \
    --z_dim Z_DIM \
    [--cpu] \
    [--n_classes N_CLASSES] \
    [--data_dim DATA_DIM] \
    [--img_size IMG_SIZE] \
    [--img_channels IMG_CHANNELS]
```



## Results



### GAN (Vanilla GAN)

<img src="./assets/gan-ring8-samples.gif" width=30% /> <img src="./assets/gan-grid25-samples.gif" width=30% /> <img src="./assets/gan-mnist-samples.gif" width=30% /> 

The **mode collapse** problem can be clearly observed.



### WGAN

<img src="./assets/wgan-ring8-samples.gif" width=30% /> <img src="./assets/wgan-grid25-samples.gif" width=30% /> <img src="./assets/wgan-mnist-samples.gif" width=30% />



### WGAN-GP

<img src="./assets/wgan-gp-ring8-samples.gif" width=30% /> <img src="./assets/wgan-gp-grid25-samples.gif" width=30% /> <img src="./assets/wgan-gp-mnist-samples.gif" width=30% />



### VEEGAN

<img src="./assets/veegan-ring8-samples.gif" width=30% /> <img src="./assets/veegan-grid25-samples.gif" width=30% /> <img src="./assets/veegan-mnist-samples.gif" width=30% />



### DCGAN

<img src="./assets/dcgan-mnist-random.png" width=40% />



### SNGAN

<img src="./assets/sngan-mnist-random.png" width=40% /> <img src="./assets/sngan-celeba-random.png" width=40% />



### LSGAN

<img src="./assets/lsgan-mnist-random.png" width=40% /> <img src="./assets/lsgan-celeba-random.png" width=40% />



### SAGAN

<img src="./assets/sagan-mnist-random.png" width=40% /> <img src="./assets/sagan-celeba-random.png" width=40% />



### CGAN

<img src="./assets/cgan-mnist-random.png" width=40% /> <img src="./assets/cgan-fashion-mnist-random.png" width=40% />



### ACGAN

<img src="./assets/acgan-mnist-random.png" width=40% /> <img src="./assets/acgan-fashion-mnist-random.png" width=40% />



### SNGAN-projection

<img src="./assets/sngan-projection-mnist-random.png" width=40% /> <img src="./assets/sngan-projection-fashion-mnist-random.png" width=40% />
