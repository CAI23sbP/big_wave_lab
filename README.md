# Big Wave for Isaac Lab Projects

## 1. Installation

```bash
python -m pip install -e source/big_wave_lab
```

## 2. Train primitive-skills

### 2.1. Squatting (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Squat-Flat-Unitree-H1-v0  --headless
```

```bash
python3 scripts/rsl_rl/train.py --task Isaac-Squat-Flat-Pro-v0  --headless
```

### 2. Reaching (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Reach-Flat-Unitree-H1-v0  --headless
```
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Reach-Flat-Pro-v0  --headless
```

### 3. Walking (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Walk-Rough-Unitree-H1-v0  --headless
```
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Walk-Rough-Pro-v0  --headless
```

## 3. Play primitive-skills

### 3.1. Squatting (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Squat-Flat-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Squat-Flat-Pro-Play-v0
```

### 3.2. Reaching (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Reach-Flat-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Reach-Flat-Pro-Play-v0
```


### 3.3. Walking (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Walk-Rough-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Walk-Rough-Pro-Play-v0
```

## 4. Train downstram task (TODO list)

### 4.1. transition (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Transition-Flat-Unitree-H1-v0 --headless
```
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Transition-Flat-Pro-v0 --headless
```


## 5. Play downstram task


### 5.1. transition (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Transition-Flat-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Transition-Pro-Play-v0
```


## 6. Train part-wise adaptation

### 6.1. pick and place (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-Place-Adaptation-Unitree-H1-v0 --headless
```
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-Place-Adaptation-Pro-v0 --headless
```


## 7. Play part-wise adaptation


### 7.1. pick and place (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-Place-Adaptation-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-Place-Adaptation-Pro-Play-v0
```


## Citation

If you use this code for your research, you must cite the following paper:

```
@article{kuang2025skillblender,
  title={SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending},
  author={Kuang, Yuxuan and Geng, Haoran and Elhafsi, Amine and Do, Tan-Dzung and Abbeel, Pieter and Malik, Jitendra and Pavone, Marco and Wang, Yue},
  journal={arXiv preprint arXiv:2506.09366},
  year={2025}
}
```