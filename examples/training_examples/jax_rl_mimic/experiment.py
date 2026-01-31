import os
import sys
os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

import pandas as pd
from pathlib import Path
import os
import loco_mujoco
import numpy as np

PROJECT_ROOT = Path(loco_mujoco.__file__).parent.parent
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "dataset/processed_embeddings")
INDEX_PATH = os.path.join(PROJECT_ROOT, "index.csv")


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



@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:

        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)


        embeddings_array, text_list = load_embeddings_from_files(
            config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path, INDEX_PATH, EMBEDDING_DIR
        )

        env.th.embeddings = embeddings_array
        env.th.text_idxs = text_list

        # get initial agent configuration
        agent_conf = PPOJax.init_agent_conf(env, config)

        # setup metric handler (optional)
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # build training function
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        # save agent state
        agent_state = out["agent_state"]
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time
        t_start = time.time()
        # get the metrics and log them
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({"Mean Episode Return": training_metrics.mean_episode_return[i],
                         "Mean Episode Length": training_metrics.mean_episode_length[i]},
                        step=int(training_metrics.max_timestep[i]))

                if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                    run.log({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                             "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                            step=int(training_metrics.max_timestep[i]))

                    # log all measures
                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    # metric for used for wandb sweep (optional)
                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                    run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                            step=int(training_metrics.max_timestep[i]))

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
                           train_state_seed=0)
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
