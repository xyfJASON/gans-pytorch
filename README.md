# GANs-Implementations

Implement GANs with PyTorch.



## Progress

**Unconditional image generation (CIFAR-10)**:

- [x] DCGAN (vanilla GAN)
- [x] DCGAN + R1 regularization
- [x] WGAN
- [x] WGAN-GP
- [x] SNGAN
- [x] LSGAN

**Conditional image generation (CIFAR-10)**:

- [x] CGAN
- [x] ACGAN

**Unsupervised decomposition (MNIST)**:

- [x] InfoGAN
- [x] EigenGAN

**Mode collapse study (Ring8, MNIST)**:

- [x] GAN (vanilla GAN)
- [x] GAN + R1 regularization
- [x] WGAN
- [x] WGAN-GP
- [x] SNGAN
- [x] LSGAN
- [x] VEEGAN

<br/>



## Unconditional Image Generation

**Notes**:

|     Model      | G. Arch.  |    D. Arch.    |                Loss                |                           Configs                            |
| :------------: | :-------: | :------------: | :--------------------------------: | :----------------------------------------------------------: |
|     DCGAN      | SimpleCNN |   SimpleCNN    |              Vanilla               |          [config file](./configs/gan_cifar10.yaml)           |
| DCGAN + R1 reg | SimpleCNN |   SimpleCNN    |   Vanilla<br/>R1 regularization    | [config file](./configs/gan_cifar10.yaml)<br/><details><summary>Additional args</summary>`--train.loss_fn.params.lambda_r1_reg 10.0`</details> |
|      WGAN      | SimpleCNN |   SimpleCNN    | Wasserstein<br/>(weight clipping)  |          [config file](./configs/wgan_cifar10.yaml)          |
|    WGAN-GP     | SimpleCNN |   SimpleCNN    | Wasserstein<br/>(gradient penalty) |        [config file](./configs/wgan_gp_cifar10.yaml)         |
|     SNGAN      | SimpleCNN | SimpleCNN (SN) |              Vanilla               |         [config file](./configs/sngan_cifar10.yaml)          |
|     SNGAN      | SimpleCNN | SimpleCNN (SN) |               Hinge                |      [config file](./configs/sngan_hinge_cifar10.yaml)       |
|     LSGAN      | SimpleCNN |   SimpleCNN    |            Least Sqaure            |         [config file](./configs/lsgan_cifar10.yaml)          |

- SN stands for "Spectral Normalization".

- For simplicity, the network architecture in all experiments is SimpleCNN, namely a stack of `nn.Conv2d` or `nn.ConvTranspose2d` layers. The results can be improved by adding more parameters and using advanced architectures (e.g., residual connections), but I decide to use the simplest setup here.

- All models except LSGAN are trained for 40k generator update steps. However, the optimizers and learning rates are not optimized for each model, so some models may not reach their optimal performance.



**Quantitative results**:

|        Model         |  FID ↓  | Inception Score ↑ |
| :------------------: | :-----: | :---------------: |
|        DCGAN         | 24.7311 |  7.0339 ± 0.0861  |
|    DCGAN + R1 reg    | 24.1535 |  7.0188 ± 0.1089  |
|         WGAN         | 49.9169 |  5.6852 ± 0.0649  |
|       WGAN-GP        | 28.7963 |  6.7241 ± 0.0784  |
| SNGAN (vanilla loss) | 24.9151 |  6.8838 ± 0.0667  |
|  SNGAN (hinge loss)  | 28.5197 |  6.7429 ± 0.0818  |
|        LSGAN         | 28.4850 |  6.7465 ± 0.0911  |

- The FID is calculated between 50k generated samples and the CIFAR-10 training split (50k images).
- The Inception Score is calculated on 50k generated samples.



**Visualization**:

<table style="text-align: center">
    <tr>
        <th>DCGAN</th>
        <th>DCGAN + R1 reg</th>
        <th>WGAN</th>
        <th>WGAN-GP</th>
    </tr>
    <tr>
        <td><img src="./assets/gan/cifar10.png"/></td>
        <td><img src="./assets/gan-r1reg/cifar10.png"/></td>
        <td><img src="./assets/wgan/cifar10.png"/></td>
        <td><img src="./assets/wgan-gp/cifar10.png"/></td>
    </tr>
    <tr>
        <th>SNGAN (vanilla loss)</th>
        <th>SNGAN (hinge loss)</th>
        <th>LSGAN</th>
    </tr>
    <tr>
        <td><img src="./assets/sngan/cifar10.png"/></td>
        <td><img src="./assets/sngan/hinge-cifar10.png"/></td>
        <td><img src="./assets/lsgan/cifar10.png"/></td>
    </tr>
</table>

<br/>



## Conditional Image Generation

**Notes**:

|   Model    | G. Arch.  | D. Arch.  | G. cond. | D. cond. |  Loss   |                 Configs & Args                 |
| :--------: | :-------: | :-------: | :------: | :------: | :-----: | :--------------------------------------------: |
|    CGAN    | SimpleCNN | SimpleCNN |  concat  |  concat  | Vanilla |   [config file](./configs/cgan_cifar10.yaml)   |
| CGAN (cBN) | SimpleCNN | SimpleCNN |   cBN    |  concat  | Vanilla | [config file](./configs/cgan_cbn_cifar10.yaml) |
|   ACGAN    | SimpleCNN | SimpleCNN |   cBN    |    AC    | Vanilla |  [config file](./configs/acgan_cifar10.yaml)   |


- cBN stands for "conditional Batch Normalization"; SN stands for "Spectral Normalization"; AC stands for "Auxiliary Classifier"; PD stands for "Projection Discriminator".



**Quantitative results**:

|   Model    |  FID ↓  |                         Intra FID ↓                          | Inception Score ↑ |
| :--------: | :-----: | :----------------------------------------------------------: | :---------------: |
|    CGAN    | 25.4999 | 47.7334<br/>            <details><summary>Details</summary><p>Class 0: 53.4163</p><p>Class 1: 44.3311</p><p>Class 2: 53.1971</p><p>Class 3: 52.2223</p><p>Class 4: 36.9577</p><p>Class 5: 65.0020</p><p>Class 6: 37.9598</p><p>Class 7: 48.3610</p><p>Class 8: 41.8075</p><p>Class 9: 44.0796</p></details> |  7.5597 ± 0.0909  |
| CGAN (cBN) | 25.3466 | 47.4136<br/>                <details><summary>Details</summary><p>Class 0: 51.5959</p><p>Class 1: 46.6855</p><p>Class 2: 49.9857</p><p>Class 3: 53.6737</p><p>Class 4: 35.1658</p><p>Class 5: 65.7719</p><p>Class 6: 38.0958</p><p>Class 7: 44.7279</p><p>Class 8: 43.3078</p><p>Class 9: 45.1265</p></details> |  7.7541 ± 0.0944  |
|   ACGAN    | 19.9154 | 49.9892<br/><details><summary>Details</summary><p>Class 0: 47.3203</p><p>Class 1: 38.6481</p><p>Class 2: 62.5885</p><p>Class 3: 66.2386</p><p>Class 4: 64.5535</p><p>Class 5: 60.7876</p><p>Class 6: 58.9524</p><p>Class 7: 36.8940</p><p>Class 8: 28.5964</p><p>Class 9: 35.3120</p></details> |  7.9903 ± 0.1038  |


- The FID is calculated between 50k generated samples (5k for each class) and the CIFAR-10 training split (50k images).
- The intra FID is calculated between 5k generated samples and CIFAR-10 training split within each class.
- The Inception Score is calculated on 50k generated samples.



**Visualizations**:

<table style="text-align: center">
    <tr>
        <th>CGAN</th>
        <th>CGAN (cBN)</th>
        <th>ACGAN</th>
    </tr>
    <tr>
        <td><img src="./assets/cgan/cifar10.png"/></td>
        <td><img src="./assets/cgan/cifar10-cbn.png"/></td>
        <td><img src="./assets/acgan/cifar10.png"/></td>
    </tr>
</table>



<br/>



## Unsupervised Decomposition

**InfoGAN**

<p align="center">
  <img src="./assets/infogan/disc.png" width=35% />
  <img src="./assets/infogan/cont.png" width=35% />
</p>

- Left: change the discrete latent variable, which corresponds to the digit type.
- Right: change one of the continuous latent variable from -1 to 1. However, the decomposition is not clear.
- Note: I found that batch normalization layers play an important role in InfoGAN. Without BN layers, the discrete latent variable tends to have a clear meaning as shown above, while the continuous variables have little effect. On the contrary, with BN layers, it's harder for the discrete variable to catch the digit type information and easier for continuous ones to find rotation in digits.

<br/>

**EigenGAN**

Random samples (no truncation):

<img src="./assets/eigengan/ffhq.png" width=100% />

Traverse:

<img src="./assets/eigengan/ffhq_traverse.png" width=100% />

<br/>



## Mode Collapse Study

Mode collapse is a notorious problem in GANs, where the model can only generate a few modes of the real data. Various methods have been proposed to solve it. To study this problem, I experimented different methods on the following two datasets:

- **Ring8**: eight gaussian distributions lying on a ring.
- **MNIST**: handwritten digit dataset.

For simplicity, the model architecture in all experiments is SimpleMLP, namely a stack of `nn.Linear` layers, thus the quality of generated MNIST image may not be so good. However, this section aims to demonstrate the mode collapse problem rather than to achieve the best image quality.

<br/>

**GAN**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>1000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/gan/ring8/step000199.png" ></td>
        <td><img src="./assets/gan/ring8/step000399.png" ></td>
        <td><img src="./assets/gan/ring8/step000599.png" ></td>
        <td><img src="./assets/gan/ring8/step000799.png" ></td>
        <td><img src="./assets/gan/ring8/step000999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>2000 steps</th>
        <th>3000 steps</th>
        <th>4000 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/gan/mnist/step000999.png" ></td>
        <td><img src="./assets/gan/mnist/step001999.png" ></td>
        <td><img src="./assets/gan/mnist/step002999.png" ></td>
        <td><img src="./assets/gan/mnist/step003999.png" ></td>
        <td><img src="./assets/gan/mnist/step004999.png" ></td>
    </tr>
</table>

On the Ring8 dataset, it can be clearly seen that all the generated data gather to only one of the 8 modes.

In the MNIST case, the generated images eventually collapse to 1.

<br/>

**GAN + R1 regularization**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/gan-r1reg/ring8/step000199.png" ></td>
        <td><img src="./assets/gan-r1reg/ring8/step000399.png" ></td>
        <td><img src="./assets/gan-r1reg/ring8/step000599.png" ></td>
        <td><img src="./assets/gan-r1reg/ring8/step000799.png" ></td>
        <td><img src="./assets/gan-r1reg/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>3000 steps</th>
        <th>5000 steps</th>
        <th>7000 steps</th>
        <th>9000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/gan-r1reg/mnist/step000999.png" ></td>
        <td><img src="./assets/gan-r1reg/mnist/step002999.png" ></td>
        <td><img src="./assets/gan-r1reg/mnist/step004999.png" ></td>
        <td><img src="./assets/gan-r1reg/mnist/step006999.png" ></td>
        <td><img src="./assets/gan-r1reg/mnist/step008999.png" ></td>
    </tr>
</table>

R1 regularization, a technique to stabilize the training process of GANs, can prevent mode collapse in vanilla GAN as well.

<br/>

**WGAN**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan/ring8/step000199.png" ></td>
        <td><img src="./assets/wgan/ring8/step000399.png" ></td>
        <td><img src="./assets/wgan/ring8/step000599.png" ></td>
        <td><img src="./assets/wgan/ring8/step000799.png" ></td>
        <td><img src="./assets/wgan/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>3000 steps</th>
        <th>5000 steps</th>
        <th>7000 steps</th>
        <th>9000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan/mnist/step000999.png" ></td>
        <td><img src="./assets/wgan/mnist/step002999.png" ></td>
        <td><img src="./assets/wgan/mnist/step004999.png" ></td>
        <td><img src="./assets/wgan/mnist/step006999.png" ></td>
        <td><img src="./assets/wgan/mnist/step008999.png" ></td>
    </tr>
</table>

WGAN indeed resolves the mode collapse problem, but converges much slower due to weight clipping.

<br/>

**WGAN-GP**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan-gp/ring8/step000199.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step000399.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step000599.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step000799.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>3000 steps</th>
        <th>5000 steps</th>
        <th>7000 steps</th>
        <th>9000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan-gp/mnist/step000999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step002999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step004999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step006999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step008999.png" ></td>
    </tr>
</table>

WGAN-GP improves WGAN by replacing the hard weight clipping with the soft gradient penalty.

The pathological weights distribution in WGAN's discriminator does not appear in WGAN-GP, as shown below.

<p style="text-align: center">
    <img src="./assets/wgan_stats.png" width=40% />
    <img src="./assets/wgan_gp_stats.png" width=40% />
</p>
<br/>

**SNGAN**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/sngan/ring8/step000199.png" ></td>
        <td><img src="./assets/sngan/ring8/step000399.png" ></td>
        <td><img src="./assets/sngan/ring8/step000599.png" ></td>
        <td><img src="./assets/sngan/ring8/step000799.png" ></td>
        <td><img src="./assets/sngan/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>3000 steps</th>
        <th>5000 steps</th>
        <th>7000 steps</th>
        <th>9000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/sngan/mnist/step000999.png" ></td>
        <td><img src="./assets/sngan/mnist/step002999.png" ></td>
        <td><img src="./assets/sngan/mnist/step004999.png" ></td>
        <td><img src="./assets/sngan/mnist/step006999.png" ></td>
        <td><img src="./assets/sngan/mnist/step008999.png" ></td>
    </tr>
</table>

Note: The above SNGAN is trained with the vanilla GAN loss instead of the hinge loss.

SNGAN uses spectral normalization to control the Lipschitz constant of the discriminator. Even with the vanilla GAN loss, SNGAN can avoid mode collapse problem.

<br/>

**LSGAN**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/lsgan/ring8/step000199.png" ></td>
        <td><img src="./assets/lsgan/ring8/step000399.png" ></td>
        <td><img src="./assets/lsgan/ring8/step000599.png" ></td>
        <td><img src="./assets/lsgan/ring8/step000799.png" ></td>
        <td><img src="./assets/lsgan/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>3000 steps</th>
        <th>5000 steps</th>
        <th>7000 steps</th>
        <th>9000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/lsgan/mnist/step000999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step002999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step004999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step006999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step008999.png" ></td>
    </tr>
</table>

LSGAN uses MSE instead of Cross-Entropy as the loss function to overcome the vanishing gradients in vanilla GAN. However, it still suffers from the mode collapse problem. For example, as shown above, LSGAN fails to cover all 8 modes on the Ring8 dataset.

Note: Contrary to the claim in the paper, I found that LSGAN w/o batch normalization does not converge on MNIST.

<br/>

**VEEGAN**

<table style="text-align: center">
    <tr>
        <th>200 steps</th>
        <th>400 steps</th>
        <th>600 steps</th>
        <th>800 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/veegan/ring8/step000199.png" ></td>
        <td><img src="./assets/veegan/ring8/step000399.png" ></td>
        <td><img src="./assets/veegan/ring8/step000599.png" ></td>
        <td><img src="./assets/veegan/ring8/step000799.png" ></td>
        <td><img src="./assets/veegan/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>3000 steps</th>
        <th>5000 steps</th>
        <th>7000 steps</th>
        <th>10000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/veegan/mnist/step000999.png" ></td>
        <td><img src="./assets/veegan/mnist/step002999.png" ></td>
        <td><img src="./assets/veegan/mnist/step004999.png" ></td>
        <td><img src="./assets/veegan/mnist/step006999.png" ></td>
        <td><img src="./assets/veegan/mnist/step009999.png" ></td>
    </tr>
</table>

VEEGAN uses an extra network to reconstruct the latent codes from the generated data.



## Run the code



### Pretrained weights

The checkpoints and training logs are stored in [xyfJASON/GANs-Implementations](https://huggingface.co/xyfJASON/GANs-Implementations/tree/main) on huggingface.



### Train

For GAN, WGAN-GP, SNGAN, LSGAN:

```shell
accelerate-launch scripts/train.py -c ./configs/xxx.yaml
```

For WGAN (weight clipping), InfoGAN, VEEGAN, CGAN, ACGAN and EigenGAN, use the scripts with corresponding name instead:

```shell
accelerate-launch scripts/train_xxxgan.py -c ./configs/xxx.yaml
```



### Sample

**Unconditional GANs**:

```shell
accelerate-launch scripts/sample.py \
    -c ./configs/xxx.yaml \
    --weights /path/to/saved/ckpt/model.pt \
    --n_samples N_SAMPLES \
    --save_dir SAVE_DIR
```

**Conditional GANs**:

```shell
accelerate-launch scripts/sample_cond.py \
    -c ./configs/xxx.yaml \
    --weights /path/to/saved/ckpt/model.pt \
    --n_classes N_CLASSES \
    --n_samples_per_class N_SAMPLES_PER_CLASS \
    --save_dir SAVE_DIR
```

**EigenGAN**:

```shell
accelerate-launch scripts/sample_eigengan.py \
    -c ./configs/xxx.yaml \
    --weights /path/to/saved/ckpt/model.pt \
    --n_samples N_SAMPLES \
    --save_dir SAVE_DIR \
    --mode MODE
```



### Evaluate

Sample images following the instructions above and use tools like [torch-fidelity](https://github.com/toshas/torch-fidelity) to calculate FID / IS.
