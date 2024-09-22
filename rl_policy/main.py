import os, re
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
    from algo import ppo  

    train_param = cfg.train.params
    is_testing = cfg.test
    ckpt_path = cfg.checkpoint
    print(f'triplet: {cfg.triplet}, objectOffset: {env.objoffset}')
    if not is_testing:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{cfg.algo}_{time_str}_s{cfg.seed}" if not cfg.exp_name else cfg.exp_name
        # triplet = re.search(r'\([^)]+\)', train_param.expertCkptFile).group(0)
        triplet = cfg.triplet  # train_param.expertCkptFiles[0].split('/')[-3]
        # verb = triplet.strip('()').split(', ')[0]
        if 'dagger' in cfg.algo:
            numtotal = len(train_param.expertCkptFiles)
            tripletlist = [re.search(r'\((.*?)\)', each).group().strip('()').split(', ')[0] for each in train_param.expertCkptFiles]
            assert len(set(tripletlist)) == 1
            log_dir = os.path.join(train_param.log_dir, train_param.expertModel, train_param.Kfuturestep, f"{tripletlist[0]}/distill{numtotal}", exp_name)
        else:
            log_dir = os.path.join(train_param.log_dir.replace('multippo', cfg.rewfunc), 'freq3/exp2', triplet, exp_name, f'{cfg.num_envs//10000}w')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
    else:
        log_dir = None

    if train_param.name == "ppo":
        runner = ppo.PPO(
            vec_env=env,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
        )
    elif train_param.name == "ippo":
        runner = ppo.IPPO(
            vec_env=env,
            train_param=train_param,
            log_dir=log_dir,
            apply_reset=False,
            record_dof=cfg['record']
        ) 
    elif train_param.name == "m3dagger":
        print(f'use {train_param.expertModel} as expert')
        from algo import dagger
        if 'IPPO' in train_param.expertModel:
            runner = dagger.M3DaggerIPPO(
                vec_env=env,
                train_param=train_param,
                expert_class=ppo.IPPO,
                log_dir=log_dir,
            )  
        elif 'PPO' in train_param.expertModel:
            runner = dagger.M3DaggerPPO(
                vec_env=env,
                train_param=train_param,
                expert_class=ppo.PPO,
                log_dir=log_dir,
            )
    else:
        raise ValueError("Unrecognized algorithm!")
    

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
        i = 1
        while True:
            # action[:, [10,11,12,13,10+22,11+22,12+22,13+22]] = 1-2*((i//10)%2)*torch.tensor([1,1,1,1,1,1,1,1,], dtype=torch.float32)
            # action[:, [10,11, 10+22,11+22]] = (1-2*((i//10)%2))*torch.tensor([-1,-1,1,1,], dtype=torch.float32)
            action[:, [10,10+22]] = (1-2*((i//10)%2))*torch.tensor([-1,1,], dtype=torch.float32)
            # action[:, [10]] = 1-2*((i//10)%2)*torch.tensor([1,], dtype=torch.float32)
            _, _, _, _ = env.step(action)
            i+=1
            if i % 100 == 0:
                print(env.robot_dof_pos)

    elif cfg.task['mode'] == "train":
        runner = build_runner(cfg, env)
        runner.run()
    elif cfg.task['mode'] == "visualize":
        env.visualize()

if __name__ == "__main__":
    main()
