# GFPack++: Improving 2D Irregular Packing by Learning Gradient Field with Attention

Official implementation of ICCV 2025 paper **GFPack++: Attention-Driven Gradient Fields for Optimizing 2D Irregular Packing**. 
This repository contains our implementation of the model using [Jittor](https://github.com/Jittor/Jittor). We will also provide a PyTorch version.

## Abstract

2D irregular packing is a classic combinatorial optimization problem with various applications, such as material utilization and texture atlas generation. This NP-hard problem requires efficient algorithms to optimize space utilization. Conventional numerical methods suffer from slow convergence and high computational cost. Existing learning-based methods, such as the score-based diffusion model, also have limitations, such as no rotation support, frequent collisions, and poor adaptability to arbitrary boundaries, and slow inferring. The difficulty of learning from teacher packing is to capture the complex geometric relationships among packing examples, which include the spatial (position, orientation) relationships of objects, their geometric features, and container boundary conditions. Representing these relationships in latent space is challenging. We propose GFPack++, an attention-based gradient field learning approach that addresses this challenge. It consists of two pivotal strategies: attention-based geometry encoding for effective feature encoding and attention-based relation encoding for learning complex relationships. We investigate the utilization distribution between the teacher and inference data and design a weighting function to prioritize tighter teacher data during training, enhancing learning effectiveness. Our diffusion model supports continuous rotation and outperforms existing methods on various datasets. We achieve higher space utilization over several widely used baselines, one-order faster than the previous diffusion-based method, and promising generalization for arbitrary boundaries. We plan to release our code and datasets to support further research in this direction.

## Environment

1. System

 - Ubuntu 20.04 or later

 - CUDA 12.4

2. Software

 - Python 3.10

 - [Jittor](https://github.com/Jittor/Jittor)

 - [JittorGeometric](https://github.com/AlgRUC/JittorGeometric)

3. Packages

We recommend using the ``environment.yml`` file to create a clean, minimal Conda environment.

```shell
conda env update --name your_env --file environment.yml
```

If you are prompted with missing dependencies when running with the minimal environment, you can use full environment with all dependencies from `environment-full.yml`:


```shell
conda env update --name your_env --file environment-full.yml
```

You can also manually supplement the packages in `environment-full.yml` .

## Dataset

Download the `dataset.pkl` file and place it in the appropriate location in the project. And in train.py, make sure the dirName variable inside the ``collectData`` and `collectValiData` function points to the correct path to the file.

```python
def collectData():
    dirName = "../dataset_dental.pkl" # dirName 为你的dataset.pkl文件地址
    dataFile = open(dirName, "rb")
	#...
    
def collectValiData():
    dirName = "../dataset_dental_vali.pkl"
    dataFile = open(dirName, "rb")
    #...
```

## How to use

Please download the pre-trained model and place it in this project directory. You can then run:

```shell
python train.py
```

You can modify the `--visualize_freq` and `--vali_freq` parameters in the arguments to adjust the frequency of visualization and validation. 

The loss and visualization results can be viewed in TensorBoard:

```shell
tensorboard --logdir=logs
```

If you encounter issues with `calutil` and `rmspacing` , you can recompile them using the following steps:

```
cd cpps
./build.sh
./build_util.sh
```

Then copy the recompiled `calutil` and `rmspacing` to the project directory.

## Citation

If you found this code useful please cite our work as:

```

```

