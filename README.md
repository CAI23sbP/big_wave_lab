# Big Wave for Isaac Lab Projects

## Installation

```bash
python -m pip install -e source/big_wave_lab
```

## training 

### 1. Squatting
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Squat-Flat-Unitree-H1-v0  --headless
```

### 2. Stepping
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Step-Flat-Unitree-H1-v0  --headless
```

### 3. Reaching
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Reach-Flat-Unitree-H1-v0  --headless
```

### 4. Walking
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Walk-Flat-Unitree-H1-v0  --headless
```


## play

### 1. Squatting

```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Squat-Flat-Unitree-H1-Play-v0
```

### 2. Stepping
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Step-Flat-Unitree-H1-Play-v0
```

### 3. Reaching
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Reach-Flat-Unitree-H1-Play-v0
```


### 4. Walking
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Walk-Flat-Unitree-H1-Play-v0
```