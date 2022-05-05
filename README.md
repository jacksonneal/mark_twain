# mark_twain

CS7150 Deep Learning Final Project

Applying Deep Learning to the [Numerai](https://numer.ai/tournament) data science competition.

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

### Demo Predictions

Load saved BASE model, load Numerai data, execute predictions.
Predictions formatted for submission to Numerai and output to ./predictions.csv in project root.

```bash
# no GPU
make demo-base
```

```bash
# use GPU
make demo-base-gpu
```

Training walk through with EDA available in [code_walkthrough.ipynb](code%20walkthrough.ipynb)

### Single Run Train Configuration

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

3. View Tensorboard logs

```bash
make tb-logs
```

### Weights and Biases Sweep Run Train Configuration

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

### Predictions From Model Checkpoint

Load saved model configs and weights to execute predictions.  File paths are relative to project root.
Run predictions with one of:

```bash
# no GPU
make predict ckpt=path/to/.ckpt hparams=path/to/.yaml
```

```bash
# use GPU
make predict-gpu ckpt=path/to/.ckpt hparams=path/to/.yaml
```

### Submit Predictions to Numerai

> __NOTE__: Submission requires api keys in local `.env` file

```bash
make submit model=(BASE|AEMLP|TMLP|CAE)
```
