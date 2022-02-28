# Efficient quantum resampling of smooth distributions

This repository contains code to reproduce the algorithm presented in ***Efficient quantum resampling of smooth distributions*** (link). The code uses the quantum simulation library `qibo` (https://github.com/qiboteam/qibo) and the author recommends enhancing its performance by installing `qibojit` (https://github.com/qiboteam/qibojit).

The `main.ipynb` file is a jupyter notebook with the steps to use QFT resampling on probability distributions and images.

The `qft_resampling_class.py` file contains the two classes to implement the QFT resampling algorithm on 1D and 2D data.