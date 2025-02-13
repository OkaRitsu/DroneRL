# DroneRL
This repo based on: https://github.com/Genesis-Embodied-AI/Genesis/tree/main/examples/drone
## Setup
https://note.com/npaka/n/n07b448c74613
https://note.com/npaka/n/n086f5e017394?sub_rt=share_pw 
```
conda create --name hover python=3.10 -y
conda activate hover
conda install pytorch torchvision torchaudio -c pytorch

pip install -r requirements.txt
```
## Run Examples
Check simulator
```
python fly.py -v -m
```

Train agent
```
python hover_train.py -e drone-hovering -B 8192 --max_iterations 500
```

Watch training log
```
tensorboard --logdir logs
```

Evaluate agent
```
python hover_eval.py -e drone-hovering --ckpt 500 --record
```