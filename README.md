# BiDexHD 🤖
🙌 Official implementation of "Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations"

## 0. Overview
we propose a unified and scalable three-phase framework BiDexHD
- Task Construction: construct a bimanual tool-using task from each demonstration in TACO dataset
- Teacher Learning: train state-based multi-task teacher policies for constructed bimanual dexterous tasks in parallel
- Student Learning: Distill teacher policies into single vision-based (pointcloud) policy

## 1. Install 🚀
### 1.1 Download isaac gym preview 📑
Refer to https://developer.nvidia.com/isaac-gym/download .

### 1.2 Download TACO dataset 🍔
Follow the instructions in https://github.com/leolyliu/TACO-Instructions .
All triplets can be found in `${dataset_root}/overall/Object_Poses/*` .

### 1.3 Build up virtual environment 🛰️
```bash
conda create -y -n bidexhd python=3.8
conda activate bidexhd
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd rl_policy/taco_dataset/
pip install -e .
cd rl_policy/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
# 
cd /home/zbh/Downloads/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
cd /home/zbh/Downloads/IsaacGymEnvs/
pip install -e .
pip install ipdb addict yapf h5py sorcery pynvml seaborn einops tensorboard accelerate open3d anytree chumpy kornia pytransform3d nlopt natsort hydra omegaconf nvitop trimesh gym git+https://github.com/isaac-sim/IsaacGymEnvs.git -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

## 2. Implement 📚
### 2.1 Task construction 🗂️
Run the code below to store all task data including pose sequences of tool, target object and hands in `rl_policy/taco_dataset/sampled_data/${triplet}.json` and visualize all tasks under ${triplet} in Isaac Gym with different object ids.  
```bash
bash vis_all.sh "'$triplet'"
```
triplet can be:
- "(empty, bowl, bowl)"
- "(stir-fry, spatula, pan)"
- ...

### 2.2 Multi-Task Reinforcement Learning 📥
Train and evaluate IPPO:
```bash
bash scripts/exe6.sh "'$triplet'" "$train_ids" 
# or parallel: bash scripts/exe6p.sh
```

### 2.3 Policy Distillation 🏆
Train and evaluate DAgger:
```bash
bash scripts/exem3dagger.sh "$device_id" "'$verb'" 
```

