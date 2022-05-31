
<div align="center">  
  
# Thermodynamics-informed graph neural networks

[![Project page](https://img.shields.io/badge/-Project%20page-blue)](https://amb.unizar.es/people/quercus-hernandez/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2203.01874.pdf)
[![IEEE TAI](https://img.shields.io/badge/IEEE%20TAI-2022-green)](https://arxiv.org/pdf/2203.01874.pdf)

</div>

## Abstract

In this paper we present a deep learning method to predict the temporal evolution of dissipative dynamic systems. We propose using both geometric and thermodynamic inductive
biases to improve accuracy and generalization of the resulting integration scheme. The first is achieved with Graph Neural
Networks, which induces a non-Euclidean geometrical prior with permutation invariant node and edge update functions. The second bias is forced by learning the GENERIC structure of the problem, an extension of the Hamiltonian formalism, to model more general non-conservative dynamics. Several examples are provided in both Eulerian and Lagrangian description in the context of fluid and solid mechanics respectively, achieving relative mean errors of less than 3% in all the tested examples. Two ablation studies are provided based on recent works in both physics-informed and geometric deep learning.

For more information, please refer to the following:

- Hernández, Quercus and Badías, Alberto and González, David and Chinesta, Francisco and Cueto, Elías. "[Thermodynamics-informed graph neural networks](https://arxiv.org/abs/2203.01874)." IEEE Transactions on Artificial Intelligence (2022).

<div align="center">
<img src="/outputs/beam.gif" width="450"><img src="/outputs/cylinder.gif" width="350">
</div>

## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/quercushernandez/ThermodynamicsGNN.git
cd ThermodynamicsGNN
```

Then, install the needed dependencies. The code is implemented in [Pytorch](https://pytorch.org). _Note that this has been tested using Python 3.9_.

```bash
# install dependencies
pip install numpy scipy matplotlib pytorch torch-geometric
 ```

## How to run the code  

### Test pretrained nets

The results of the paper (Couette flow, bending beam and flow past a cylinder) can be reproduced with the following scripts, found in the `executables/` folder.

```bash
python main.py --sys_name couette --train False --n_hidden 2 --dim_hidden 10 --passes 10
python main.py --sys_name beam --train False --n_hidden 2 --dim_hidden 50 --passes 10
python main.py --sys_name cylinder --train False --n_hidden 2 --dim_hidden 128 --passes 10
```

The `data/` folder includes the database and the pretrained parameters of the networks. The resulting time evolution of the state variables is plotted and saved in .gif format in a generated `outputs/` folder.

|             Couette Flow                  |         Bending Beam                     |             Flow past a Cylinder          |
| ------------------------------------------|------------------------------------------| ------------------------------------------|
|<div align="center"> <img src="/data/couette.png" width="250"></div>|<div align="center"> <img src="/data/beam.png" width="250"></div>| <div align="center"> <img src="/data/cylinder.png" width="250"></div> |

### Train a custom net

You can also run your own experiments for the implemented datasets by setting custom parameters manually. Several training examples can be found in the `executables/` folder. The manually trained parameters and output plots are saved in the `outputs/` folder.

```bash
e.g.
python main.py --sys_name beam --train True --lr 1e-3 ...
```

General Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--sys_name`              | Study case                                        | `couette`, `beam`, `cylinder`                         |
| `--train`                 | Train mode                                        | `True`, `False`                                       |
| `--gpu`                   | Enable GPU acceleration                           | `True`, `False`                                       |
| `--output_dir`            | Output data directory                             | Default: `output`                                     |
| `--plot_sim`              | Plot a test simulation                            | `True`, `False`                                       |

Training Arguments:

|     Argument              |             Description                           | Options                                               |
|---------------------------| ------------------------------------------------- |------------------------------------------------------ |
| `--n_hidden`              | Number of MLP hidden layers                       | Default: `2`                                          |
| `--dim_hidden`            | Dimension of hidden layers                        | Default: `10`                                         |
| `--passes`                | Number of message passing blocks                  | Default: `10`                                         |
| `--lr`                    | Learning rate                                     | Default: `1e-3`                                       |
| `--lambda_d`              | Data loss weight                                  | Default: `1e2`                                        |
| `--noise_var`             | Variance of the training noise                    | Default: `1e-2`                                       |
| `--batch_size`            | Training batch size                               | Default: `2`                                          |
| `--max_epoch`             | Maximum number of training epochs                 | Default: `6000`                                       |
| `--miles`                 | Learning rate scheduler milestones                | Default: `2000 4000`                                  |
| `--gamma`                 | Learning rate scheduler decay                     | Default: `1e-1`                                       |

## Citation

If you found this code useful please cite our work as:

```
@article{hern2020structurepreserving,
    title={Thermodynamics-informed graph neural networks},
    author={Quercus Hernandez and Alberto Badias and Francisco Chinesta and Elias Cueto},
    year={2020},
    eprint={2004.04653},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
