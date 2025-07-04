# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING
from .g1_cfg import G1_CFG


from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""
    
    # reward
    # rew_termination = -0.0              # 终止惩罚，避免早期死亡
    # rew_action_l2 = -5              # 动作惩罚，稍微减小
    # rew_joint_pos_limits = -1         # 关节限制惩罚，加大避免违规
    # rew_joint_acc_l2 = -0.000001          # 关节加速度惩罚，鼓励平滑动作
    # rew_joint_vel_l2 = -0.005           # 关节速度惩罚，稍微减小
    rew_termination = -0.0
    rew_action_l2 = -0.1
    rew_joint_pos_limits = -10
    rew_joint_acc_l2 = -1.0e-06
    rew_joint_vel_l2 = -0.001
    # imitation reward parameters
    rew_imitation_pos: float = 1.0       # 位置误差奖励：重要，整体轨迹
    rew_imitation_rot: float = 0.5
    rew_imitation_joint_pos: float = 2.5
    rew_imitation_joint_vel: float = 1.0  # 从2.0降到1.0，减小权重避免过度惩罚
    imitation_sigma_pos: float = 1.2      # 从0.8调到1.2，配合分段奖励函数
    imitation_sigma_rot: float = 0.5      # 从1.0调到1.4，给角度误差更大容忍度
    imitation_sigma_joint_pos: float = 1.5 # 从1.2调到1.5，关节位置容忍度
    imitation_sigma_joint_vel: float = 8.0 # 从2.5大幅调到8.0，应对300-500的高误差
    # env
    episode_length_s = 10.0
    decimation = 1

    # spaces
    observation_space =  71 + 3 * (8+5) - 6 + 1  # 加入 progress 特征
    action_space = 29
    state_space = 0
    num_amp_observations = 3
    amp_observation_space = 71 + 3 * (8 + 5) - 6 + 1

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random-start"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_dance.npz")
    
@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")