# mark_twain

CS7150 Deep Learning Final Project

Code authors
------------
Jackson Neal  
Rohit Barve  
Quay Dragon

GPU Access
------------
> __NOTE__: GPU configuration is optional but recommended for compute intensive jobs.

1. SSH to `login.discovery.neu.edu`
2. Request GPU partition

```bash
make req-gpu
```

3. Load anaconda and cuda modules

```bash
module load anaconda3/2022.01
```

```bash
module load cuda/11.1
```

4. Activate *pytorch_env_training* environment
```bash
source activate pytorch_env_training
````

Installation
------------

```bash
make init
```

Execution
------------

### Single Run Configuration

1. Modify single run hyperparameters in [single.yaml](numerai/config/single.yaml)
2. Run trainer with one of:

```bash
# no GPU
make single
```

```bash
# use GPU
make single-gpu
```

3. View Tensorboard logs in `lightning_logs`

```bash
make tb-logs
```

### Weights and Biases Sweep Run Configuration

> __NOTE__: Executing sweep runs requires an api key

1. Modify sweep run hyperparameters in [sweep.yaml](numerai/config/sweep.yaml)
2. Run trainer with on of:

```bash
# no GPU
make sweep
```

```bash
# use GPU
make sweep-gpu
```

3. View sweep results at [https://wandb.ai/cs7150-jn](https://wandb.ai/cs7150-jn)
