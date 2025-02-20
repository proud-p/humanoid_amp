#!/usr/bin/env python
import argparse
import time
import numpy as np
import torch

# ========= 解析命令行参数 ==========
parser = argparse.ArgumentParser(
    description="Potential Field Controller Demo in Isaac Lab (Repulsive Only)."
)
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# 这里添加 AppLauncher 相关的命令行参数
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ========= 启动 Omniverse 应用 ==========
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ========= 导入 Isaac Lab 相关模块 ==========
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip G1_MINIMAL_CFG
from isaaclab_assets import G1_MINIMAL_CFG  

# ========= 定义桌面场景配置 ==========
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """桌面场景的配置"""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # 根据命令行选择机器人模型，目前只支持 franka_panda（加载的是 G1_MINIMAL_CFG）
    if args_cli.robot == "franka_panda":
        robot = G1_MINIMAL_CFG.replace(prim_path="/World/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")

# ========= 辅助函数：打印机器人资产下所有 prim 的路径 ==========
def list_robot_prims():
    try:
        # 使用 USD Python API 遍历 "/World/Robot" 下的所有子节点
        from pxr import Usd
        stage = Usd.Stage.GetCurrent()
        robot_prim = stage.GetPrimAtPath("/World/Robot")
        if robot_prim:
            print("在 '/World/Robot' 下发现的 prim 路径：")
            for prim in robot_prim.GetChildren():
                print(prim.GetPath())
        else:
            print("在 '/World/Robot' 未找到任何 prim，请检查机器人是否正确加载。")
    except Exception as e:
        print("获取 USD Stage 时出错：", e)

# ========= 运行仿真 ==========
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """主循环：运行仿真并打印出机器人关节与刚体信息（请先观察打印结果确定名称规则）"""
    
    # 先打印出机器人资产下所有 prim 路径，帮助你确认实际的命名规则
    list_robot_prims()
    
    # 如果你在上面的输出中找到了关节和身体对应的命名，可以在下面设置相应的正则表达式。
    # 例如，假设你观察后发现关节名称中包含 "joint" 而刚体名称中包含 "body"，可以这样设置：
    robot = scene["robot"]
    
    print("解析到的关节名称列表:", robot.joint_names)
    print("解析到的刚体名称列表:", robot.body_names)

    step_count = 0
    while simulation_app.is_running():
        # 更新场景并执行一步物理仿真
        scene.update(dt=sim.get_physics_dt())
        scene.write_data_to_sim()
        sim.step()
        step_count += 1

# ========= 主函数 ==========
def main():
    # 创建仿真配置，指定仿真步长和设备（例如 "cuda:0" 或 "cpu"）
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # 初始化场景（由 InteractiveScene 管理所有实体）
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置仿真环境
    sim.reset()
    print("[INFO]: Setup complete. Starting simulation...")

    # 进入主仿真循环
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
