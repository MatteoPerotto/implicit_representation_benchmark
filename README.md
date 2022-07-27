# implicit_representation_benchmark

## Installation
You can install the conda environment by running
```
conda env create -f env.yml
```
Then install the metrics by running:
```
pip install metrics/chamfer_dist metrics/chamfer_dist
```
For the installation to be successful the system cuda version must match the one usedd by pytorch (`11.3` if you installed from `env.yml`)

## Running the code
To execute the code you have two options:
 - run one of the configurations in `configs`. They specify the parameters to use and the experiment to run (recommended)
 - run one of the scripts in `experiments`. To specify the config to be used you can modify `configs/__init__.py`
