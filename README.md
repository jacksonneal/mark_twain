# mark_twain

CS7150 Deep Learning Final Project

Code author
------------
Jackson Neal

GPU Access
------------
> __NOTE__: GPU configuration is optional but may be necessary for compute intensive jobs.

1. SSH to `login.discovery.neu.edu`
2. Activate *pytorch_env_training* environment

```bash
> srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=08:00:00 /bin/bash
> module load anaconda3/2022.01
> module load cuda/11.1
> source activate pytorch_env_training
```

Installation
------------

```bash
> pip install -r requirements.txt
```

Execution
------------

### Single Run Configuration

1. Modify single run hyperparameters in [single.yaml](single.yaml)
2. Run trainer with one of:

```bash
# no GPU
> make single

# use GPU
> make single-gpu
```

3. View Tensorboard logs in [lightning_logs](lightning_logs)

```bash
> make tb-logs
```

### Weights and Biases Sweep Run Configuration

> __NOTE__: Executing sweep runs requires an api key

1. Modify sweep run hyperparameters in [sweep.yaml](sweep.yaml)
2. Run trainer with on of:

```bash
# no GPU
> make sweep

# use GPU
> make sweep-gpu
```

3. View sweep results at [https://wandb.ai/cs7150-jn](https://wandb.ai/cs7150-jn)