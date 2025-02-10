# Data Augmentation and Regularization for Learning Group Equivariance

This repository contains codes used in the experiments of the paper

```
@article{nordenfors2025augmentation
  title={Data Augmentation and Regularization for Learning Group Equivariance},
  author={Nordenfors, Oskar and Flinth, Axel},
  journal={In preparation},
  year={2025}
}
```

It is mainly released for transparency and reproducibility purposes. Should you find this code useful for your research, please cite the above paper.

## Required and optional libraries
To train the models, [pytorch](https://pytorch.org/) (with torchvision to handle MNIST), [numpy](https://numpy.org/) and  [os](https://docs.python.org/3/library/os.html) are needed.  [tqdm](https://tqdm.github.io/) is used for generating progress bars. For plotting,  [matplotlib](https://matplotlib.org/) is used.

## Running experiments

To train 30 models with $\gamma=10^0$, run the script

```
  python train_networks.py 0 30
```

The script will do 30 runs, just as in the paper, in an unparallelized fashion. Running the script with arguments `n m` will instead set $\gamma=10^n$ and run m runs.

To subsequently plot the results, run 

```
  python train_networks.py 0 30
```

with obvious adjustments if you have used a different value of $\gamma$ and a different number of runs.
