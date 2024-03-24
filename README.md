# Inverse PINN - steady state heat equation

This project is an implementation of solving an inverse problem using the Physics-informed neural network (PINN). The problem to be solved is the steady-state heat equation,

$-\nabla \cdot h(x,y)\nabla T = q$

where the thermal conductivity $h(x,y)$ is spatially varying and an unknown term that is being identified.

More details are on the implementation are included in the `report.pdf`

## Installation

This project has been implemented in Python `3.12.2` using Numpy, PyTorch and matplotlib. Use the package manager [pip](https://pip.pypa.io/en/stable/) to run this project.

```bash
pip install -r requirements.txt
```
To install PyTorch with CUDA, refer to [PyTorch](https://pytorch.org/get-started/locally/) website for instructions.

## Usage

To run the training script, run the `train.py`. An example command would be:
```bash
python train.py --train_data ../dataset/data_nodim.csv --in_col 1 2 --out_col 0 --iter 1000
```
To view and tune other training hyperparameters and problem setup, use `python train.py --help` to view arguments.

To visualize the inference results from the function, run this command:
``` bash
python plotting_result.py
```
By default the exact solution of $h(x,y)$ is set to $1 + 6x^2 + \frac{x}{1+2y^2}$ but the exact solution can be changed using `--exact` options, for example to input exact solution as $h(x,y)=x+y$, the command would be as follows:
``` bash
python plotting_result.py --exact x+y
```
## Contact
If you have any questions about the implementation, please contact me at mttrung94@gmail.com