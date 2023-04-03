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

### Find best configuration for mixed-precision matmul, using GA
Run the `main_phase2.py` program with desired arguments:
```
python3 main_phase2.py \
	--model_file fc_net.h5 \
	--layer_idx -1 \
	--max_bit_constraint 2500 \
	--pop_size 50 \
	--num_generations 30 \
	--crossover_rate 0.25 \
	--mutation_rate 0.01 \
	--output_file best.csv \
	--log_file ga.log
```

The first three arguments are for inputs of the maximization problem and
should always be passed to. `--max_bit_constraint` is left to the runner to
decide, and its default value is arbitrary.

The four GA parameters (`--pop_size`, `--num_generations`, `--crossover_rate`
and `--mutation_rate`) are optional and come with sensible defaults.

If passed to, the best solution, in form of mixed-precision matrix, will be
written to `--output_file` in csv.

If passed to, log will be saved at `--log_file`.
