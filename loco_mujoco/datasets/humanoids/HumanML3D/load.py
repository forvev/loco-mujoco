import pandas as pd
import os
import numpy as np
import os
from pathlib import Path
import loco_mujoco

PROJECT_ROOT = Path(loco_mujoco.__file__).parent.parent
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "dataset/processed_embeddings")
INDEX_PATH = os.path.join(PROJECT_ROOT, "index.csv")


def load_embedding_from_file(dataset_path: str) -> tuple:
    """
    Loads specific .npy embeddings for the requested dataset_paths.
    """
    # Extract the dataset name from the path
    dataset_name = os.path.basename(dataset_path).replace(".npz", "")

    try:
        df = pd.read_csv(INDEX_PATH)

        # We strip spaces and ensure string types for safety
        source_to_id = dict(
            zip(df["source_path"].astype(str).str.strip(), df["new_name"].astype(str))
        )

        embedding = None
        text_idx = None

        found_id = None

        for source, hid in source_to_id.items():
            if dataset_name in source:
                found_id = hid.split(".")[0]  # "000321.npy" -> "000321"
                break

        if not found_id:
            print(f"[WARN] ID not found for: {dataset_name}. Using Zero Vector.")
            embedding = np.zeros(768)
            return embedding, text_idx

        # Load the specific .npy file
        npy_path = os.path.join(EMBEDDING_DIR, f"{found_id}_embed.npy")

        if os.path.exists(npy_path):
            emb = np.load(npy_path)
            embedding = emb
            text_idx = found_id  # TODO: we can change to text if needed
        else:
            print(f"[WARN] Embedding file missing: {npy_path}")
            embedding = np.zeros(768)

    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return np.array([]), []

    return embedding, text_idx
