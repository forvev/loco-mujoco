from typing import Union, ClassVar, List
from dataclasses import dataclass

from loco_mujoco.trajectory.dataclasses import Trajectory


@dataclass
class DefaultDatasetConf:
    """
    Configuration for loading default datasets provided by LocoMuJoCo.

    Attributes:
        dataset_type (str): The type of the dataset to load. Can be "mocap" or "pretrained".
        task (str): The task to load.
        debug (bool): Whether to load the dataset in debug mode.

    """

    task: Union[str, list]  = "walk"
    dataset_type: str = "mocap"
    debug: bool = True

    def __post_init__(self):
        assert self.dataset_type in ["mocap", "pretrained"], f"Unknown dataset type: {self.dataset_type}"


@dataclass
class AMASSDatasetConf:
    """
    Configuration for loading AMASS datasets.

    Attributes:
        rel_dataset_path (Union[str, list]): A relative path or a list of relative paths to
            load from the AMASS dataset.
        dataset_group (str): A name of a predefined group of datasets to load from AMASS.

    """
    rel_dataset_path: Union[str, list] = None
    dataset_group: str = None

    def __post_init__(self):
        assert self.rel_dataset_path is not None or self.dataset_group is not None, ("Either `rel_dataset_path` or "
                                                                                     "`dataset_group` must be set.")


@dataclass
class LAFAN1DatasetConf:
    """
    Configuration for loading LAFAN1 datasets.

    Attributes:
        dataset_name (Union[str, list]): A name of a dataset or a list of dataset names to load from LAFAN1.
        dataset_group (str): A name of a predefined group of datasets to load from LAFAN1.

    ..note:: This datatset is loaded from the LocoMuJoCo's HuggingFace repository:
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets. It provides datasets for
        all humanoid environments.

    """

    dataset_name: Union[str, list] = None
    dataset_group: str = None

    def __post_init__(self):
        assert self.dataset_name is not None or self.dataset_group is not None, ("Either `dataset_name` or "
                                                                                 "`dataset_group` must be set.")


@dataclass
class HumanML3DDatasetConf:
    """
    Configuration for loading HumanML3D datasets via text prompts.

    Attributes:
        prompts (Union[str, list]): A text prompt or list of prompts to retrieve motions for.
        dataset_path (str): Path to the HumanML3D root directory (must contain 'texts' and 'amass_data').
    """

    prompts: Union[str, list] = None

    def __post_init__(self):
        assert self.prompts is not None, (
            "`prompts` must be set."
        )


@dataclass
class CustomDatasetConf:
    """
    Configuration for loading custom trajectories.

    Attributes:
        traj (Trajectory): A custom trajectory to load.

    """
    traj: Trajectory
