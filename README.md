ln -s ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp ~/IsaacLab/

ln -s ~/IsaacLab/scripts/reinforcement_learning/skrl ~/IsaacLab/

Train:
./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-Humanoid-AMP-Run-Direct-v0 --headless

./isaaclab.sh -p ~/IsaacLab/skrl/train.py --task Isaac-G1-AMP-Dance-Direct-v0 --headless

Eval：
./isaaclab.sh -p ~/IsaacLab/skrl/play.py --task Isaac-Humanoid-AMP-Run-Direct-v0 --num_envs 32 

TensorBoard:
./isaaclab.sh -p -m tensorboard.main --logdir logs/skrl/