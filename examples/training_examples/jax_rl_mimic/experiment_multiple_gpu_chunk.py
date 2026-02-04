import os
import sys
# for testing
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
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
from omegaconf import DictConfig, OmegaConf, open_dict
import traceback
import dataclasses

@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    print(f"Number of devices: {jax.local_device_count()}")
    
    # --- 1. SETUP WANDB (ONCE) ---
    # We init Wandb outside the loop so it tracks the whole 'lifetime' as one run
    wandb.login()
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    run = wandb.init(project=config.wandb.project, config=config_dict, resume="allow")
    
    # --- 2. PREPARE THE DATA CHUNKS ---
    # Retrieve the full list of files from your config
    full_dataset_list = list(config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path) * 6000
    
    # USER SETTING: How many files to load at once?
    # You found this number in your stress test (e.g. 500 or 1000)
    CHUNK_SIZE = 100
    
    # If the list is short (e.g. 1 file), we might want to repeat it to simulate a long run
    # Uncomment the next line if you are testing with just 1 file but want 10 loops
    # full_dataset_list = full_dataset_list * 10 

    # Create the chunks
    # Result: [['file1', 'file2'], ['file3', 'file4'], ...]
    dataset_chunks = [full_dataset_list[i:i + CHUNK_SIZE] for i in range(0, len(full_dataset_list), CHUNK_SIZE)]
    
    print(f"==================================================")
    print(f" RELAY TRAINING INITIALIZED")
    print(f" Total Files: {len(full_dataset_list)}")
    print(f" Chunk Size:  {CHUNK_SIZE}")
    print(f" Total Loops: {len(dataset_chunks)}")
    print(f"==================================================\n")

    # --- 3. STATE VARIABLES ---
    latest_agent_state = None  # Holds the "Brain"
    global_step_offset = 0     # Keeps Wandb x-axis continuous
    
    # --- 4. THE CHUNKING LOOP ---
    for chunk_idx, current_chunk_files in enumerate(dataset_chunks):
        print(f"\n>>> STARTING CHUNK {chunk_idx + 1}/{len(dataset_chunks)}")
        print(f"    Loading {len(current_chunk_files)} files...")
        
        try:
            # A. CONFIGURE CURRENT CHUNK
            with open_dict(config):
                # Overwrite the dataset path with just this chunk's files
                config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = current_chunk_files
                
                # OPTIONAL: Recalculate num_updates if needed
                # If you want a fixed number of updates per chunk, set it here:
                # config.experiment.num_updates = 500 

            # B. BUILD ENVIRONMENT (Memory Spike 1)
            factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
            env = factory.make(
                **config.experiment.env_params, **config.experiment.task_factory.params
            )


            if latest_agent_state is None:
                agent_conf = PPOJax.init_agent_conf(env, config)
            else:
                agent_conf = latest_agent_state # Resume
            # agent_conf = PPOJax.init_agent_conf(env, config)
            # if latest_agent_state is not None:
            #     print("    Resuming Agent: Transplanting trained weights...")
            #     # Now this works because you added the 'initial_train_state' field!
            #     agent_conf = dataclasses.replace(agent_conf, initial_train_state=latest_agent_state)
            # else:
            #     print("    Initializing NEW Agent (Random Weights)...")
            #     pass
            # if latest_agent_state is not None:
            #     agent_conf = dataclasses.replace(agent_conf, train_state=latest_agent_state)
            #     print("    Initializing NEW Agent...")
            #     agent_conf = PPOJax.init_agent_conf(env, config)
            # else:
            #     # Subsequent Loops: Inject the trained brain
            #     print("    Resuming Agent from previous chunk...")
            #     # In PPOJax, the 'agent_conf' is structurally identical to the 'agent_state'
            #     # So we can simply pass the saved state as the configuration
            #     agent_conf = latest_agent_state

            # D. COMPILE TRAINING FUNCTION (Memory Spike 2)
            mh = MetricsHandler(config, env) if config.experiment.validation.active else None
            train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

            if config.experiment.n_seeds > 1:
                train_fn = jax.pmap(train_fn, axis_name="devices")
            else:
                train_fn = jax.jit(train_fn)

            # E. RUN TRAINING
            print("    Compiling & Training...")
            # Generate RNG keys
            rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
            rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
            
            out = train_fn(_rng)
        

            # F. SAVE THE BRAIN (For the next loop)
            # Extract the state. If using pmap, we get shape (N_Devices, ...), so take index 0
            raw_state = out["agent_state"]
            if config.experiment.n_seeds > 1:
                latest_agent_state = jax.tree.map(lambda x: x[0], raw_state)
            else:
                latest_agent_state = raw_state
            
            # Optional: Save to disk every loop just in case
            PPOJax.save_agent(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, agent_conf, latest_agent_state)

            # G. LOG METRICS (Continuous)
            if not config.experiment.debug:
                print("    Logging metrics...")
                training_metrics = jax.device_get(out["training_metrics"])
                # Handle averaging across devices/seeds
                if config.experiment.n_seeds > 1:
                     training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
                
                # Log to Wandb
                max_step_in_this_run = 0
                for i in range(len(training_metrics.mean_episode_return)):
                    current_step = int(training_metrics.max_timestep[i]) + global_step_offset
                    max_step_in_this_run = max(max_step_in_this_run, int(training_metrics.max_timestep[i]))
                    
                    run.log({
                        "Mean Episode Return": training_metrics.mean_episode_return[i],
                        "Mean Episode Length": training_metrics.mean_episode_length[i],
                    }, step=current_step)
                
                # Update the global offset for the next loop
                global_step_offset += max_step_in_this_run
                print(f"    Chunk Finished. Global Step is now: {global_step_offset}")

            # H. CLEANUP (CRITICAL FOR 2TB)
            # print("    Cleaning up memory...")
            # del env, train_fn, out, agent_conf, factory, raw_state, mh
            # jax.clear_caches()
            # gc.collect()
            # print("    Memory Cleared.")

        except Exception as e:
            print(f"\nXXX CHUNK {chunk_idx} FAILED XXX")
            traceback.print_exc()
            # Optional: break or continue? Usually break to debug.
            break

    # --- 5. FINISH ---
    print("\n=== ALL CHUNKS COMPLETED ===")
    wandb.finish()

if __name__ == "__main__":
    experiment()