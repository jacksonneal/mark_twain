# mark_twain

CS7150 Deep Learning Final Project


Code author
------------
Jackson Neal

Required
------------
- Python 3.9

Install
------------
```bash
> pip install -r requirements.txt
```

Execution
---------
ssh [user]@login.discovery.neu.edu

srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=08:00:00 /bin/bash
