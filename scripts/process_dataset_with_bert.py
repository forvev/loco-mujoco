import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.BERT.BERT_encoder import load_bert
from pathlib import Path
import loco_mujoco

# Paths (Adjust to your structure)
PROJECT_ROOT = Path(loco_mujoco.__file__).parent.parent
INDEX_PATH = os.path.join(PROJECT_ROOT, "index.csv")
TEXTS_DIR = os.path.join(PROJECT_ROOT, "dataset/HumanML3D/texts")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset/processed_embeddings")


def load_bert_model(device):
    print("Loading BERT model...")
    # This downloads 'bert-base-uncased' from HuggingFace automatically
    bert = load_bert("bert-base-uncased")
    bert.text_model.to(device)
    bert.eval()  # Set to evaluation mode (no dropout, etc.)
    return bert


def process_dataset():
    # 1. Load Index Mapping
    if not os.path.exists(INDEX_PATH):
        print(f"Error: {INDEX_PATH} not found.")
        return

    df = pd.read_csv(INDEX_PATH)

    # We use a set to avoid processing duplicates
    unique_ids = set()

    # Filter for KIT files
    for _, row in df.iterrows():
        source = str(row["source_path"])
        if "KIT" in source:
            # new_name is usually "000123.npy" -> extract "000123"
            file_id = str(row["new_name"]).split(".")[0]
            unique_ids.add(file_id)

    print(f"Found {len(unique_ids)} unique KIT motions in index.")

    # 2. Initialize BERT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    bert = load_bert_model(device)

    # 3. Process
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for file_id in tqdm(list(unique_ids), desc="Processing Embeddings"):
        text_path = os.path.join(TEXTS_DIR, f"{file_id}.txt")

        # A. Extract Text
        if not os.path.exists(text_path):
            skipped_count += 1
            continue

        caption = None
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                # Iterate lines to find the first valid HumanML3D formatted line
                for line in f:
                    line = line.strip()
                    if "#" in line:
                        # Format: caption#tokens#start#end
                        caption = line.split("#")[0]
                        break
        except Exception as e:
            print(f"Error reading {text_path}: {e}")
            skipped_count += 1
            continue

        if not caption:
            skipped_count += 1
            continue

        with torch.no_grad():
            feat, mask = bert([caption])

            embedding = feat[0, 0, :].cpu().numpy()

        save_path = os.path.join(OUTPUT_DIR, f"{file_id}_embed.npy")
        np.save(save_path, embedding)

        processed_count += 1

    print(f"\nProcessing Complete.")
    print(f"Successfully saved: {processed_count}")
    print(f"Skipped (missing text/errors): {skipped_count}")
    print(f"Output Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_dataset()
