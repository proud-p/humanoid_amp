### Symbolic Links
```
ln -s ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp ~/IsaacLab/

ln -s ~/IsaacLab/scripts/reinforcement_learning/skrl ~/IsaacLab/
```

### Resources

[![Demo Video](https://img.shields.io/badge/Demo-Bilibili-ff69b4?style=for-the-badge&logo=bilibili)](https://www.bilibili.com/video/BV19cRvYhEL8/?vd_source=5159ce41348cd4fd3d83ef9169dc8dbc)
[![Documentation](https://img.shields.io/badge/Documentation-DeepWiki-blue?style=for-the-badge&logo=gitbook)](https://deepwiki.com/linden713/humanoid_amp)

### Motions Scripts
- `motion_loader.py` - Load motion data from npz files and provide sampling functionality
- `motion_viewer.py` - 3D visualization player for motion data
- `data_convert.py` - Convert CSV motion data to npz format with interpolation and forward kinematics
- `motion_replayer.py` - Replay motion data in Isaac Sim with optional recording
- `record_data.py` - Recording and managing motion data utility classes
- `verify_motion.py` - Verify and display npz file contents
- `visualize_motion.py` - Generate interactive HTML charts to visualize motion data

### Train
```
./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-G1-AMP-Walk-Direct-v0 --headless
```
or
```
./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-G1-AMP-Dance-Direct-v0 --headless
```
### Eval
```
./isaaclab.sh -p ~/IsaacLab/skrl/play.py --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 32 
```
### TensorBoard
```
./isaaclab.sh -p -m tensorboard.main --logdir logs/skrl/
```

The parameters of the code in this repository have not been fine-tuned. Currently, the walk performance is acceptable, but the dance performance is quite poor. Due to personal bussiness, I will not begin to debug until summer.
### Dataset & URDF
~~The dataset and URDF files are from [Hugging Face Unitree Robotics](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset).~~

**Note**: The original dataset and URDF files from [Unitree Robotics](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset) have been removed by the official source.

If you're still looking for the dataset, a third-party mirror is currently available here:  
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

*Use at your own discretion, as it is not officially maintained. qwq*

**Contributions**, **discussions**, and stars are all welcome! ❥(^_-)
