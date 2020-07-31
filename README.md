# Semi-supervised-VAE
Semi-supervised VAE model implementation and experiment.
## Introduction

This repo contains several implementations of semi-supervised models based on Variational Auto-encoder (VAE).
These models are listed below:

+ Conditional VAE (CVAE) which includes label info in the model
+ Stacked CVAE
+ Gussian-mixture VAE

The experiment is conducted on MNIST dataset.



## Requirements

+ python3.7
+ pytorch
+ torchvision



## Training and experiments

Run

```bash
bash ./scripts/train.sh


```

to train the model.

After training is complete, run

```bash
bash ./scripts/exp.sh
```

to do the experiments.

## License
MIT
