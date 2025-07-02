![Humanoid AMP](docs/human_amp.png)

---

# Humanoid AMP
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacSim](https://img.shields.io/badge/IsaacLab-2.1.0-silver.svg
)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


## Symbolic Links
```
ln -s ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp ~/IsaacLab/

ln -s ~/IsaacLab/scripts/reinforcement_learning/skrl ~/IsaacLab/
```
## Train
```
./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-G1-AMP-Walk-Direct-v0 --headless
```
or
```
./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-G1-AMP-Dance-Direct-v0 --headless
```
## Eval
```
./isaaclab.sh -p ~/IsaacLab/skrl/play.py --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 32 
```
## TensorBoard
```
./isaaclab.sh -p -m tensorboard.main --logdir logs/skrl/
```

Developing on the dev branch.
The usage of some helper script functions is explained at the beginning of the file.

## Dataset & URDF

**Note**: The original dataset and URDF files from [Unitree Robotics](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset) have been removed by the official source.

If you're still looking for the dataset, a third-party mirror is currently available here:  
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

Or you can also use the data from [AMASS](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)
## TODO

- ✅ Current: Dancing motion is working
- [ ] Test the `test` branch thoroughly and merge it into `main` as soon as possible
- [ ] Write a more detailed README to cover the new features and usage
- [ ] Add clearer comments and explanations in all Python scripts

## Others
Video: [Bilibili](https://www.bilibili.com/video/BV19cRvYhEL8/?vd_source=5159ce41348cd4fd3d83ef9169dc8dbc)

DeepWiki: [humanoid_amp](https://deepwiki.com/linden713/humanoid_amp)

**Contributions**, **discussions**, and **stars** are all welcome! ❥(^_-)
