# Neural network with mixed-precision matrix-multiplication

## Prepare virtual environment

### With `conda`
```bash
conda env create -f environment.yml
conda activate precision
```

### With `python-virtualenv`
Use `requirements.txt`

## Run

### Training
First, to train a simple fully-connected (FC) neural net on MNIST dataset, run:
```
python3 train.py
```
The model will be saved to a file called `fc_net.h5`.

### Inference with mixed-precision matrix-multiplication
```
python3 main_phase1.py
```

This program makes the last FC layer of the neural net to use mixed-precision
in its mat-mul operation. The program looks for a mixed-precision matrix in a
file named `mpmat.csv`. If this file does not exists, the program will randomly
create one.
