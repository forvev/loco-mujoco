import os
import sys

########### for testing ###############
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = (
#     "--xla_force_host_platform_device_count=2 "
#     "--xla_gpu_triton_gemm_any=True"
# )
#######################################
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
# by default JAX will preallocate 75% of the total GPU memory when the first JAX
# operation is run.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# to load mode than 9GB of data
os.environ["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "-1"
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


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:
        print(f"Number of devices: {jax.local_device_count()}")
        print(f"Devices: {jax.devices()}")
        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # create env
        env = factory.make(
            **config.experiment.env_params, **config.experiment.task_factory.params
        )

        # get initial agent configuration
        agent_conf = PPOJax.init_agent_conf(env, config)

        # setup metric handler (optional)
        mh = (
            MetricsHandler(config, env) if config.experiment.validation.active else None
        )

        # build training function
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

        n_devices = jax.local_device_count()
        # pmap training function
        if config.experiment.n_seeds > 1:
            if config.experiment.n_seeds != n_devices:
                print(
                    f"WARNING: You requested {config.experiment.n_seeds} seeds but have {n_devices} devices."
                )
                print(
                    "ensure n_seeds equals the number of available GPUs for pmap to work 1-to-1."
                )

            # Use pmap to parallelize across devices
            train_fn = jax.pmap(train_fn, axis_name="devices")
        else:
            # Fallback for single seed/device
            train_fn = jax.jit(train_fn)

        # get rng keys and run training
        rngs = [
            jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)
        ]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        # save agent state
        agent_state = out["agent_state"]
        if config.experiment.n_seeds > 1:
            agent_state_single = jax.tree.map(lambda x: x[0], agent_state)
        else:
            agent_state_single = agent_state
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state_single)
        run.config.update({"agent_save_path": save_path})

        import time

        t_start = time.time()
        # get the metrics and log them
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # calculate mean across seeds
            training_metrics = jax.tree.map(
                lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics
            )
            validation_metrics = jax.tree.map(
                lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics
            )
            training_metrics = jax.tree.map(
                lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics
            )
            validation_metrics = jax.tree.map(
                lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics
            )

            # Iterating through a JAX array element-by-element in a Python loop is SLOW
            # because it triggers a transfer for every single number.
            # We move everything to CPU (numpy) once, then iterate.
            training_metrics = jax.device_get(training_metrics)
            validation_metrics = jax.device_get(validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log(
                    {
                        "Mean Episode Return": training_metrics.mean_episode_return[i],
                        "Mean Episode Length": training_metrics.mean_episode_length[i],
                    },
                    step=int(training_metrics.max_timestep[i]),
                )

                if (
                    (i + 1) % config.experiment.validation_interval == 0
                    and config.experiment.validation.active
                ):
                    run.log(
                        {
                            "Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[
                                i
                            ],
                            "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[
                                i
                            ],
                        },
                        step=int(training_metrics.max_timestep[i]),
                    )

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
                                    metrics_to_log[
                                        f"Validation Measures/{measure_name}/{attr_name}"
                                    ] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    # metric for used for wandb sweep (optional)
                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                    run.log(
                        {"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                        step=int(training_metrics.max_timestep[i]),
                    )

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained agent to record video
        PPOJax.play_policy(
            env,
            agent_conf,
            agent_state,
            deterministic=True,
            n_steps=200,
            n_envs=20,
            record=True,
            train_state_seed=0,
        )
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
