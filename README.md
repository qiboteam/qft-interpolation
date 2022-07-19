# Efficient quantum interpolation of smooth distributions

This repository contains code to reproduce the algorithm presented in ***Efficient quantum interpolation of natural data*** (https://arxiv.org/abs/2203.06196). The code uses the quantum simulation library `qibo` (https://github.com/qiboteam/qibo) and the we recommend enhancing its performance by installing `qibojit` (https://github.com/qiboteam/qibojit).

The `qft_inteprolation.ipynb` file is a jupyter notebook with the steps to use QFT resampling on probability distributions and images.

The `dct_inteprolation.ipynb` file is a jupyter notebook with the steps to use DCT resampling on images.

The `qft_class.py` file contains the two classes to implement the QFT resampling algorithm on 1D and 2D data.

The `qjpeg_class.py` file contains the classes to implement the DCT resampling and compression algorithm on 2D data.
