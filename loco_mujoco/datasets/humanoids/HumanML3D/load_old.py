import os
import joblib
import numpy as np
from loco_mujoco.smpl.retargeting import (
    load_amass_data, 
    fit_smpl_motion, 
    load_robot_conf_file, 
    get_smpl_model_path,
    extend_motion,
    fit_smpl_shape,
    OPTIMIZED_SHAPE_FILE_NAME
)
from loco_mujoco.utils import setup_logger

def load_humanml3d_entry(
    env_name: str,          # e.g., "UnitreeH1.walk"
    amass_file_path: str,   # Path to the specific .npz file (from HumanML3D mapping)
    output_path: str = None # Optional: Save the result immediately
):
    """
    Loads and retargets a single HumanML3D entry (AMASS file) to a robot.
    """
    logger = setup_logger("humanml3d_loader")

    # 1. Load Configuration & Paths
    # We load the specific robot configuration (e.g. UnitreeH1)
    robot_conf = load_robot_conf_file(env_name)
    smpl_model_path = get_smpl_model_path()

    # 2. Load the Raw SMPL Data
    # HumanML3D maps text to AMASS files. We load the raw AMASS data here.
    if not os.path.exists(amass_file_path):
        logger.error(f"AMASS file not found: {amass_file_path}")
        return None

    motion_data = load_amass_data(amass_file_path) 

    # 3. Check/Create Robot Shape
    # Ensures the robot's physical dimensions (like leg length) match the motion data
    robot_shape_path = f"./robot_shapes/{env_name}/{OPTIMIZED_SHAPE_FILE_NAME}"
    if not os.path.exists(robot_shape_path):
        logger.info(f"Optimizing robot shape for {env_name}...")
        fit_smpl_shape(env_name, robot_conf, smpl_model_path, robot_shape_path, logger)

    # 4. Run the Retargeting Optimization
    # Solves the IK to map SMPL poses to Robot joint angles
    traj = fit_smpl_motion(
        env_name=env_name,
        robot_conf=robot_conf,
        path_to_smpl_model=smpl_model_path,
        motion_data=motion_data,
        path_to_optimized_smpl_shape=robot_shape_path,
        logger=logger,
        skip_steps=True  # Set to False for higher precision at the cost of speed
    )

    # 5. Extend Trajectory (Kinematics)
    # Calculates velocities and other Mujoco-specific data
    traj = extend_motion(env_name, robot_conf.env_params, traj, logger)

    # 6. Save or Return
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        traj.save(output_path)
        logger.info(f"Saved robot trajectory to: {output_path}")

    return traj
