# BVDex
bimanual dexterous manipulation from videos


(download isaac gym preview) 
conda create -y -n bvdex python=3.8
conda activate bvdex
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd /home/zbh/Downloads/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
cd /home/zbh/Downloads/IsaacGymEnvs/
pip install -e .
cd /home/zbh/Desktop/zbh/robot/dex-retargeting/
pip install -e .
cd /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
pip install ipdb addict yapf h5py sorcery pynvml seaborn einops tensorboard accelerate open3d anytree chumpy pytransform3d nlopt natsort hydra omegaconf gym -i https://pypi.tuna.tsinghua.edu.cn/simple  # git+https://github.com/isaac-sim/IsaacGymEnvs.git 

```bash
python taco_dataset/TACOdataset.py --mode make_dataset --triplet "(stir, spoon, pan)"
```
triplet can be:
- "(empty, bowl, bowl)"
- "(stir-fry, spatula, pan)"

```bash
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo
```
