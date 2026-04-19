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

### 2.2. Reaching (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Reach-Flat-Unitree-H1-v0  --headless
```
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Reach-Flat-Pro-v0  --headless
```

### 2.3. Walking (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Walk-Rough-Unitree-H1-v0  --headless
```
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Walk-Rough-Pro-v0  --headless
```

### 2.4. Heading (Tienkung Pro Only)
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Head-Flat-Pro-v0  --headless
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

### 3.4. Heading (Tienkung Pro Only)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Head-Flat-Pro-Play-v0
```

## 4. Train downstram task (TODO list)

### 4.1. pick and place (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-and-Place-Adaptation-Unitree-H1-v0 --headless
```
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-and-Place-Adaptation-Pro-v0 --headless
```

## 5. Play downstram task

### 5.1. pick and place (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-and-Place-Adaptation-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-and-Place-Adaptation-Pro-Play-v0
```


## 6. Train vision based downstream task (For sim2real)

### 6.1. pick and place vision (H1-2, Tienkung Pro)
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-and-Place-Adaptation-Vision-Unitree-H1-v0 --headless
```
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-and-Place-Adaptation-Vision-Pro-v0 --headless
```


## 7. Play vision based downstream task (For sim2real)

### 7.1. pick and place vision (H1-2, Tienkung Pro)

```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-and-Place-Adaptation-Vision-Unitree-H1-Play-v0
```
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-and-Place-Adaptation-Vision-Pro-Play-v0
```


## 8. Visualize Control
```bash
Press 1 or 2: Move to surroundings

Press 8: Move camera forward

Press 4: Move camera left

Press 6: Move camera right

Press 5: Move camera backward

Press 0: Use free camera (mouse enabled)

Press 1: Do not use free camera (default)
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

## Contact us

```
1. sbp0783@hanyang.ac.kr

2. snp0783@naver.com
```