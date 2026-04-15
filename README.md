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

## 4. Train downstram task

### 4.1. transition
```bash
python3 scripts/rsl_rl/train.py --task Isaac-Transition-Flat-Unitree-H1-v0 --headless
```


## 5. Play downstram task


### 5.1. transition
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Transition-Flat-Unitree-H1-Play-v0
```


## 6. Train part-wise adaptation

### 6.1. pick and place
```bash
python3 scripts/rsl_rl/train.py  --task Isaac-Pick-Place-Adaptation-Unitree-H1-v0
```


## 7. Play part-wise adaptation


### 7.1. pick and place
```bash
python3 scripts/rsl_rl/play.py  --task Isaac-Pick-Place-Adaptation-Unitree-H1-Play-v0
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


```
@inproceedings{bae2025plt,
  author    = {Bae, Jinseok and Lee, Younghwan and Lim, Donggeun and Kim, Young Min},
  title     = {PLT: Part-Wise Latent Tokens as Adaptable Motion Priors for Physically Simulated Characters},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages     = {1--10},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3721238.3730637}
}
```


```
@article{park2026maskadapt,
  author  = {Park, Soomin and Lee, Eunseong and Lee, Kwang Bin and Lee, Sung-Hee},
  title   = {MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters},
  journal = {arXiv preprint arXiv:2603.29272},
  year    = {2026}
}
```
