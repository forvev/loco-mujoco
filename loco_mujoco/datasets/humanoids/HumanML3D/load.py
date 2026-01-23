import os
import pandas as pd
import numpy as np
from pathlib import Path
import loco_mujoco
from difflib import get_close_matches
from loco_mujoco.smpl.retargeting import (
    load_amass_data,
    fit_smpl_motion,
    load_robot_conf_file,
    get_smpl_model_path,
    extend_motion,
    fit_smpl_shape,
    OPTIMIZED_SHAPE_FILE_NAME,
)
from loco_mujoco.utils import setup_logger
from loco_mujoco.smpl.retargeting import get_amass_dataset_path

logger = setup_logger("humanml3d_loader")


class HumanML3DIndex:
    """
    Handles the mapping from HumanML3D IDs (014597) to real AMASS paths.
    Format: ./pose_data/.../source.npy, start, end, 014597.npy
    """

    def __init__(self):
        package_path = Path(loco_mujoco.__file__).resolve().parent
        project_root = package_path.parent
        self.index_path = os.path.join(project_root, "index.csv")
        self.mapping = {}

        if os.path.exists(self.index_path):
            self._build_index()
        else:
            logger.warning(
                f"index.csv not found at {self.index_path}. Mapping will fail."
            )

    def _build_index(
        self,
    ):
        """ """
        try:
            df = pd.read_csv(self.index_path)

            col_source = "source_path"
            col_start = "start_frame"
            col_end = "end_frame"
            col_name = "new_name"

            for _, row in df.iterrows():
                raw_source = str(row[col_source])

                if "KIT" not in raw_source:
                    continue

                # Extract ID: "014597.npy" -> "014597"
                file_id = str(row[col_name]).split(".")[0]

                # Clean source path
                raw_source = str(row[col_source])
                if raw_source.startswith("./pose_data/"):
                    clean_source = raw_source.replace("./pose_data/", "")
                elif raw_source.startswith("pose_data/"):
                    clean_source = raw_source.replace("pose_data/", "")
                else:
                    clean_source = raw_source

                # Fix extension: Index says .npy, but AMASS is usually .npz
                if clean_source.endswith(".npy"):
                    clean_source = clean_source[:-4] + ".npz"

                self.mapping[file_id] = {
                    "rel_path": clean_source,
                    "start": int(row[col_start]),
                    "end": int(row[col_end]),
                }
        except Exception as e:
            logger.error(f"Failed to parse index.csv: {e}")

    def get_info(self, file_id: int) -> dict:
        return self.mapping.get(file_id)


def get_real_amass_path(file_id: int, index_handler: HumanML3DIndex) -> tuple:
    """
    Resolves '014597' -> '/full/path/to/BMLmovi/.../Subject_71_F_18_poses.npz'
    """
    info = index_handler.get_info(file_id)

    rel_path = info["rel_path"]
    amass_dataset_path = get_amass_dataset_path()


    full_path = os.path.join(amass_dataset_path, rel_path)
    if os.path.exists(full_path):
        return full_path, info["start"], info["end"]

    logger.error(f"File not found: {rel_path}")
    return None, 0, 0


def load_humanml3d_text_mapping() -> dict:
    """
    Reads the texts folder and builds a mapping from text prompts to HumanML3D file IDs.
    returns:
        text_mapping (dict): A dictionary mapping text prompts to HumanML3D file IDs.
    """
    package_path = Path(loco_mujoco.__file__).resolve().parent
    project_root = package_path.parent
    text_mapping = {}
    texts_dir = os.path.join(project_root, "dataset/HumanML3D/texts")

    if not os.path.exists(texts_dir):
        return {}

    # parser: Read the first line of every text file
    for filename in os.listdir(texts_dir):
        if filename.endswith(".txt"):
            file_id = filename.split(".")[0]
            try:
                with open(os.path.join(texts_dir, filename), "r") as f:
                    line = f.readline().strip()
                    # HumanML3D format: "description#tokens#start#end"
                    clean_text = line.split("#")[0] if "#" in line else line
                    text_mapping[clean_text.lower()] = file_id
            except Exception:
                continue
    return text_mapping


def load_humanml3d_by_prompt(prompt: str) -> str:
    text_map = load_humanml3d_text_mapping()
    index_handler = HumanML3DIndex()

    # 2. Find File ID
    prompt_key = prompt.lower()
    if prompt_key in text_map:
        file_id = text_map[prompt_key]
        logger.info(f"Found match: '{prompt}' -> ID: {file_id}")
    else:
        matches = get_close_matches(prompt_key, text_map.keys(), n=1, cutoff=0.4)
        if matches:
            file_id = text_map[matches[0]]
            logger.info(f"Using closest match: '{matches[0]}' -> ID: {file_id}")
        else:
            logger.error(f"No prompt match for: {prompt}")
            return None

    # 3. Resolve Path
    amass_path, start, end = get_real_amass_path(file_id, index_handler)

    if not amass_path:
        return None

    if "KIT" in amass_path:
        start = amass_path.find("KIT")
        end = amass_path.rfind(".")
        amass_path = amass_path[start:end]
    print(f"amass_path: {amass_path}")

    return amass_path


def load_humanml3d_entry(
    env_name, amass_file_path, start_frame=0, end_frame=-1, output_path=None
):
    logger.info(f"Retargeting file: {amass_file_path}")

    robot_conf = load_robot_conf_file(env_name)
    smpl_model_path = get_smpl_model_path()

    # Load Data
    if "KIT" in amass_file_path:
        start = amass_file_path.find("KIT")
        end = amass_file_path.rfind(".")
        amass_file_path = amass_file_path[start:end]
    print(f"amass_file_path: {amass_file_path}")
    motion_data = load_amass_data(amass_file_path)

    # Slice Data based on Index
    if end_frame > start_frame:
        logger.info(f"Slicing frames {start_frame} to {end_frame}")
        # Slice the main pose data
        if len(motion_data["pose_aa"]) > end_frame:
            motion_data["pose_aa"] = motion_data["pose_aa"][start_frame:end_frame]
            motion_data["trans"] = motion_data["trans"][start_frame:end_frame]
        else:
            # Handle edge case where index frame count > actual file frames
            logger.warning(
                f"Index asks for frame {end_frame} but file only has {len(motion_data['pose_aa'])}. Clipping."
            )
            motion_data["pose_aa"] = motion_data["pose_aa"][start_frame:]
            motion_data["trans"] = motion_data["trans"][start_frame:]

    # Optimize Shape
    # robot_shape_path = f"./amass_conv/{env_name}/{OPTIMIZED_SHAPE_FILE_NAME}"
    robot_shape_path = (
        "/home/zelik/projects/thesis/amass_conv/UnitreeH1/shape_optimized.pkl"
    )
    if not os.path.exists(robot_shape_path):
        logger.error(f"Optimized shape file not found: {robot_shape_path}")
        # fit_smpl_shape(env_name, robot_conf, smpl_model_path, robot_shape_path, logger)

    print(f"smpl_model_path: {smpl_model_path}")
    # Retarget
    traj = fit_smpl_motion(
        env_name=env_name,
        robot_conf=robot_conf,
        path_to_smpl_model=smpl_model_path,
        motion_data=motion_data,
        path_to_optimized_smpl_shape=robot_shape_path,
        logger=logger,
    )

    traj = extend_motion(env_name, robot_conf.env_params, traj, logger)

    if output_path:
        traj.save(output_path)

    return traj
