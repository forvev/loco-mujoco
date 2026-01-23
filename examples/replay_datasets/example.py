import numpy as np
import os

from loco_mujoco.task_factories import (
    ImitationFactory,
    LAFAN1DatasetConf,
    DefaultDatasetConf,
    AMASSDatasetConf,
    HumanML3DDatasetConf,
)


# # example --> you can add as many datasets as you want in the lists!
env = ImitationFactory.make(
    "UnitreeH1",
    # default_dataset_conf=DefaultDatasetConf(["squat", "walk"]),
    # lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
    humanml3d_dataset_conf=HumanML3DDatasetConf(
        prompts=["turn right and jump"],
        # prompts=["walking faster"],
    ),
    # if SMPL and AMASS are installed, you can use the following:
    # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
    #                                     "KIT/12/WalkInClockwiseCircle11_poses",
    #                                     "HUMAN4D/HUMAN4D/Subject3_Medhi/INF_JumpingJack_S3_01_poses",
    #                                     'KIT/359/walking_fast05_poses']),
    # amass_dataset_conf=AMASSDatasetConf([
    #                                     "KIT/314/turn_right05_poses"]),
    n_substeps=20,
)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
