import numpy as np
from loco_mujoco.task_factories import (
    ImitationFactory,
    LAFAN1DatasetConf,
    DefaultDatasetConf,
    AMASSDatasetConf,
)
import pandas as pd
from pathlib import Path
import os
import loco_mujoco

PROJECT_ROOT = Path(loco_mujoco.__file__).parent.parent
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "dataset/processed_embeddings")
INDEX_PATH = os.path.join(PROJECT_ROOT, "index.csv")

# TODO: make it more scalable
dataset_paths = [
    "KIT/378/push_recovery_stand_left03_poses",
    "KIT/9/step_over_gap05_poses",
    # "KIT/3/kick_high_left02_poses",
    "KIT/3/jump_left02_poses",
]


def load_embeddings_from_files(
    dataset_paths: list, index_path: str, embedding_dir: str
) -> tuple:
    """
    Loads specific .npy embeddings for the requested dataset_paths.
    """
    print(f"Loading embeddings for {len(dataset_paths)} motions...")

    try:
        df = pd.read_csv(index_path)

        # We strip spaces and ensure string types for safety
        source_to_id = dict(
            zip(df["source_path"].astype(str).str.strip(), df["new_name"].astype(str))
        )

        embeddings = []
        texts = []

        for path in dataset_paths:
            found_id = None
            for source, hid in source_to_id.items():
                if path in source:
                    found_id = hid.split(".")[0]  # "000321.npy" -> "000321"
                    break

            if not found_id:
                print(f"[WARN] ID not found for: {path}. Using Zero Vector.")
                embeddings.append(np.zeros(768))
                texts.append("Unknown")
                continue

            # Load the specific .npy file
            npy_path = os.path.join(embedding_dir, f"{found_id}_embed.npy")

            if os.path.exists(npy_path):
                emb = np.load(npy_path)
                embeddings.append(emb)
                texts.append(str(found_id))  # TODO: we can change to text if needed
            else:
                print(f"[WARN] Embedding file missing: {npy_path}")
                embeddings.append(np.zeros(768))
                texts.append("Missing")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return np.array([]), []

    return np.array(embeddings), texts


embeddings_array, text_list = load_embeddings_from_files(
    dataset_paths, INDEX_PATH, EMBEDDING_DIR
)

env = ImitationFactory.make(
    "UnitreeH1", amass_dataset_conf=AMASSDatasetConf(dataset_paths), n_substeps=20
)
env.th.embeddings = embeddings_array
env.th.text_idxs = text_list

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
