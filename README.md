# BVDex
bimanual dexterous manipulation from videos


(download isaac gym preview) 
cd IsaacGym_Preview_4_Package/isaacgym/python
conda create -y -n bvdex python=3.8
conda activate bvdex
pip install addict yapf h5py sorcery pynvml seaborn numpy==1.20 einops tensorboard accelerate open3d anytree pinocchio pytransform3d nlopt natsort

python -m taco_dataset.TACOdataset --mode make_dataset --triplet "(empty, bowl, bowl)"
python -m policy.train
