import os
import json
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *

import tasks


def build_runner(cfg, env):
    from algo import ppo  # , dagger, dagger_value

    train_param = cfg.train.params
    is_testing = cfg.test  # train_param["test"]
    is_vision = False
    ckpt_path = cfg.checkpoint

    if not is_testing:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(train_param.log_dir, f"{cfg.algo}_{time_str}_s{cfg.seed}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
    else:
        log_dir = None

    if train_param.name == "ppo":
        trainer_class = ppo.PPO
    elif train_param.name == "ippo":
        trainer_class = ppo.IPPO
    else:
        raise ValueError("Unrecognized algorithm!")
    print(f"Using {trainer_class.__name__} for training")
    runner = trainer_class(
        vec_env=env,
        actor_critic_class=ppo.ActorCritic,
        train_param=train_param,
        log_dir=log_dir,
        apply_reset=False,
        is_vision=is_vision,
    )

    if is_testing and ckpt_path != "":
        print(f"Loading model from {ckpt_path}")
        runner.test(ckpt_path)
    elif ckpt_path != "":
        print(f"\nWarning: load pre-trained policy. Loading model from {ckpt_path}\n")
        runner.load(ckpt_path)

    return runner


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="config")
def main(cfg: DictConfig) -> None:
    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.num_envs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{cfg.task_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env = create_isaacgym_env()
    
    if cfg.task['mode'] == "debug":
        action = torch.zeros((env.num_envs, 22*2))
        i = 0
        while True:
            # action[:, [10,11,12,13,10+22,11+22,12+22,13+22]] = 1-2*((i//10)%2)*torch.tensor([1,1,1,1,1,1,1,1,], dtype=torch.float32)
            # action[:, [10,11, 10+22,11+22]] = (1-2*((i//10)%2))*torch.tensor([-1,-1,1,1,], dtype=torch.float32)
            # action[:, [10,10+22]] = (1-2*((i//10)%2))*torch.tensor([-1,1,], dtype=torch.float32)
            # action[:, [10]] = 1-2*((i//10)%2)*torch.tensor([1,], dtype=torch.float32)
            _, _, _, _ = env.step(action)
            i+=1

    elif cfg.task['mode'] == "train":
        runner = build_runner(cfg, env)
        runner.run()
    elif cfg.task['mode'] == "visualize":
        env.visualize()

if __name__ == "__main__":
    main()
