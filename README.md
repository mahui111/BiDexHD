# BVDex
bimanual dexterous manipulation from videos


(download isaac gym preview) 
cd IsaacGym_Preview_4_Package/isaacgym/python
conda create -y -n bvdex python=3.8
conda activate bvdex
pip install ipdb addict yapf h5py sorcery pynvml seaborn numpy==1.21 einops tensorboard accelerate open3d anytree chumpy pytransform3d nlopt natsort hydra omegaconf gym git+https://github.com/isaac-sim/IsaacGymEnvs.git

```bash
python taco_dataset/TACOdataset.py --mode make_dataset --triplet "(stir, spatula, pan)"
```
triplet can be:
- "(empty, bowl, bowl)"
- "(stir-fry, spatula, pan)"

```bash
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo
```
