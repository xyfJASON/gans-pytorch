# GANs-Implementations

My implementations of GANs with PyTorch.



## Progress

- [x] GAN (vanilla GAN)
- [ ] CGAN
- [x] DCGAN
- [x] WGAN
- [x] WGAN-GP
- [x] SNGAN
- [ ] SNGAN-projection
- [ ] ACGAN
- [ ] SAGAN
- [x] LSGAN
- [ ] VEEGAN

<br/>



## Image Generation



### CIFAR-10

**Quantitative results**:

- The FID is calculated between 50k generated samples and the CIFAR-10 training split (50k images).
- The Inception Score is calculated on 50k generated samples.
- All models except LSGAN are trained for 40k generator update steps. However, the optimizers and learning rates are different for each model, so some models may not reach their optimal performance.
  - LSGAN is trained for 30k generator update steps because I found the image quality drops between 30k-40k steps.


<table style="text-align: center">
    <tr>
        <th>Model</th>
        <th>Arch</th>
        <th>Loss</th>
        <th>FID@50k ↓</th>
        <th>Inception Score ↑</th>
    </tr>
    <tr>
        <td>DCGAN</td>
        <td>Simple CNN</td>
        <td>Vanilla</td>
        <td>24.8453</td>
        <td>7.1121 ± 0.0690</td>
    </tr>
    <tr>
        <td>WGAN</td>
        <td>Simple CNN</td>
        <td>Wasserstein<br/>(weight clipping)</td>
        <td>51.0953</td>
        <td>5.5291 ± 0.0621</td>
    </tr>
    <tr>
        <td>WGAN-GP</td>
        <td>Simple CNN</td>
        <td>Wasserstein<br/>(gradient penalty)</td>
        <td>31.8783</td>
        <td>6.7391 ± 0.0814</td>
    </tr>
    <tr>
        <td>SNGAN</td>
        <td>Simple CNN</td>
        <td>Vanilla</td>
        <td>27.2486</td>
        <td>6.9839 ± 0.0774</td>
    </tr>
    <tr>
        <td>SNGAN</td>
        <td>Simple CNN</td>
        <td>Hinge</td>
        <td>28.7972</td>
        <td>6.8365 ± 0.0787</td>
    </tr>
    <tr>
        <td>LSGAN</td>
        <td>Simple CNN</td>
        <td>Least Sqaure</td>
        <td>35.2344</td>
        <td>6.3496 ± 0.0748</td>
    </tr>
</table>

**Visualization**:

<table style="text-align: center">
    <tr>
        <th>DCGAN</th>
        <th>WGAN</th>
        <th>WGAN-GP</th>
    </tr>
    <tr>
        <td><img src="./assets/dcgan-cifar10.png"/></td>
        <td><img src="./assets/wgan-cifar10.png"/></td>
        <td><img src="./assets/wgan-gp-cifar10.png"/></td>
    </tr>
    <tr>
        <th>SNGAN (vanilla loss)</th>
        <th>SNGAN (hinge loss)</th>
        <th>LSGAN</th>
    </tr>
    <tr>
        <td><img src="./assets/sngan-cifar10.png"/></td>
        <td><img src="./assets/sngan-hinge-cifar10.png"/></td>
        <td><img src="./assets/lsgan-cifar10.png"/></td>
    </tr>
</table>



<br/>



## Mode Collapse Study

Mode collapse is a notorious problem in GANs, where the model can only generate a few modes of the real data. Various methods have been proposed to solve it. To study this problem, I experimented different methods on the following two datasets:

- **Ring8**: eight gaussian distributions lying on a ring.
- **MNIST**: handwritten digit dataset.

For simplicity, the model architecture in all experiments is SimpleMLP, namely a stack of `nn.Linear` layers, thus the quality of generated MNIST image may not be so good. However, this section aims to demonstrate the mode collapse problem rather than to achieve the best image quality.



### GAN

<table style="text-align: center">
    <tr>
        <th>1000 steps</th>
        <th>2000 steps</th>
        <th>3000 steps</th>
        <th>4000 steps</th>
        <th>5000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/gan/ring8/step000999.png" ></td>
        <td><img src="./assets/gan/ring8/step001999.png" ></td>
        <td><img src="./assets/gan/ring8/step002999.png" ></td>
        <td><img src="./assets/gan/ring8/step003999.png" ></td>
        <td><img src="./assets/gan/ring8/step004999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>6000 steps</th>
        <th>12000 steps</th>
        <th>18000 steps</th>
        <th>24000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/gan/mnist/step005999.png" ></td>
        <td><img src="./assets/gan/mnist/step011999.png" ></td>
        <td><img src="./assets/gan/mnist/step017999.png" ></td>
        <td><img src="./assets/gan/mnist/step023999.png" ></td>
        <td><img src="./assets/gan/mnist/step049999.png" ></td>
    </tr>
</table>

On the Ring8 dataset, it can be clearly seen that all the generated data gather to only one of the 8 modes.

In the MNIST case, the generated images eventually collapse to 1.



### WGAN (weight clipping)

<table style="text-align: center">
    <tr>
        <th>4000 steps</th>
        <th>8000 steps</th>
        <th>12000 steps</th>
        <th>16000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan/ring8/step003999.png" ></td>
        <td><img src="./assets/wgan/ring8/step007999.png" ></td>
        <td><img src="./assets/wgan/ring8/step011999.png" ></td>
        <td><img src="./assets/wgan/ring8/step015999.png" ></td>
        <td><img src="./assets/wgan/ring8/step049999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>6000 steps</th>
        <th>12000 steps</th>
        <th>18000 steps</th>
        <th>24000 steps</th>
        <th>100000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan/mnist/step005999.png" ></td>
        <td><img src="./assets/wgan/mnist/step011999.png" ></td>
        <td><img src="./assets/wgan/mnist/step017999.png" ></td>
        <td><img src="./assets/wgan/mnist/step023999.png" ></td>
        <td><img src="./assets/wgan/mnist/step099999.png" ></td>
    </tr>
</table>

WGAN indeed resolves the mode collapse problem, but converges much slower due to weight clipping.



### WGAN-GP

<table style="text-align: center">
    <tr>
        <th>3000 steps</th>
        <th>6000 steps</th>
        <th>9000 steps</th>
        <th>12000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan-gp/ring8/step002999.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step005999.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step008999.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step011999.png" ></td>
        <td><img src="./assets/wgan-gp/ring8/step049999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>4000 steps</th>
        <th>8000 steps</th>
        <th>12000 steps</th>
        <th>16000 steps</th>
        <th>100000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/wgan-gp/mnist/step003999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step007999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step011999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step015999.png" ></td>
        <td><img src="./assets/wgan-gp/mnist/step099999.png" ></td>
    </tr>
</table>
WGAN-GP improves WGAN by replacing the hard weight clipping with the soft gradient penalty.

The pathological weights distribution in WGAN's discriminator does not appear in WGAN-GP, as shown below.

<p style="text-align: center">
    <img src="./assets/wgan_stats.png" width=40% />
    <img src="./assets/wgan_gp_stats.png" width=40% />
</p>


### SNGAN

<table style="text-align: center">
    <tr>
        <th>3000 steps</th>
        <th>6000 steps</th>
        <th>9000 steps</th>
        <th>12000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/sngan/ring8/step002999.png" ></td>
        <td><img src="./assets/sngan/ring8/step005999.png" ></td>
        <td><img src="./assets/sngan/ring8/step008999.png" ></td>
        <td><img src="./assets/sngan/ring8/step011999.png" ></td>
        <td><img src="./assets/sngan/ring8/step049999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>4000 steps</th>
        <th>8000 steps</th>
        <th>12000 steps</th>
        <th>16000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/sngan/mnist/step003999.png" ></td>
        <td><img src="./assets/sngan/mnist/step007999.png" ></td>
        <td><img src="./assets/sngan/mnist/step011999.png" ></td>
        <td><img src="./assets/sngan/mnist/step015999.png" ></td>
        <td><img src="./assets/sngan/mnist/step049999.png" ></td>
    </tr>
</table>
Note: The above SNGAN are trained with the vanilla GAN loss instead of the hinge loss.

SNGAN uses spectral normalization to control the Lipschitz constant of the discriminator. Even with the vanilla GAN loss, SNGAN can avoid mode collapse problem.



### LSGAN

<table style="text-align: center">
    <tr>
        <th>3000 steps</th>
        <th>6000 steps</th>
        <th>9000 steps</th>
        <th>12000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/lsgan/ring8/step002999.png" ></td>
        <td><img src="./assets/lsgan/ring8/step005999.png" ></td>
        <td><img src="./assets/lsgan/ring8/step008999.png" ></td>
        <td><img src="./assets/lsgan/ring8/step011999.png" ></td>
        <td><img src="./assets/lsgan/ring8/step049999.png" ></td>
    </tr>
</table>

<table style="text-align: center">
    <tr>
        <th>4000 steps</th>
        <th>8000 steps</th>
        <th>12000 steps</th>
        <th>16000 steps</th>
        <th>50000 steps</th>
    </tr>
    <tr>
        <td><img src="./assets/lsgan/mnist/step003999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step007999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step011999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step015999.png" ></td>
        <td><img src="./assets/lsgan/mnist/step049999.png" ></td>
    </tr>
</table>

LSGAN uses MSE instead of Cross-Entropy as the loss function to overcome the vanishing gradients in vanilla GAN. However, it still suffers from the mode collapse problem. For example, as shown above, LSGAN fails to cover all 8 modes on the Ring8 dataset.

Note: Contrary to the claim in the paper, I found that LSGAN w/o batch normalization does not converge on MNIST.
