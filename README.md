# Big Wave for Isaac Lab Projects

## 1. Installation

```bash
python -m pip install -e source/big_wave_lab
```

## 2. Train primitive-skills

### 2.1. Squatting
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Squat-Flat-Unitree-H1-v0  --headless
```

### 2. Reaching
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Reach-Flat-Unitree-H1-v0  --headless
```

### 3. Walking
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Walk-Rough-Unitree-H1-v0  --headless
```

## 3. Play primitive-skills

### 3. 1. Squatting

```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Squat-Flat-Unitree-H1-Play-v0
```

### 3.2. Reaching
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Reach-Flat-Unitree-H1-Play-v0
```

### 3.3. Walking
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Walk-Rough-Unitree-H1-Play-v0
```

## 4. Train high-level

```bash
python3 scripts/rsl_rl/train.py  --task Isaac-High-Level-Unitree-H1-v0
```


## 5. Play high-level


```bash
python3 scripts/rsl_rl/play.py  --task Isaac-High-Level-Unitree-H1-Play-v0
```
