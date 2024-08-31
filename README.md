# BVDex
bimanual dexterous manipulation from videos


(download isaac gym preview) 
conda create -y -n bvdex python=3.8
conda activate bvdex
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd rl_policy/taco_dataset/
pip install -e .
cd rl_policy/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd /home/zbh/Downloads/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
cd /home/zbh/Downloads/IsaacGymEnvs/
pip install -e .
pip install ipdb addict yapf h5py sorcery pynvml seaborn einops tensorboard accelerate open3d anytree chumpy kornia pytransform3d nlopt natsort hydra omegaconf nvitop trimesh gym -i https://pypi.tuna.tsinghua.edu.cn/simple  # git+https://github.com/isaac-sim/IsaacGymEnvs.git 

```bash
python taco_dataset/TACOdataset.py --mode make_dataset --triplet "(stir, spoon, pan)"
```
triplet can be:
- "(empty, bowl, bowl)"
- "(stir-fry, spatula, pan)"

```bash
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo
```

replace:
"gpos": \[.*\n.*\n.*\n.*\n.*   -> "gpos": [0.0,0.0,0.0]

(empty, kettle, cup)            x
(skim off, bowl, bowl)          x
(dust, roller, bowl)            x
(empty, cup, bowl)              x left-right-inverse
(brush, brush, cup)             x
(skim off, spoon, bowl)         x
(pour in some, bowl, cup)       x left-right-inverse
(put in, spoon, bowl)           x
(screw, screwdriver, box)       x
(put in, spoon, pan)            x
(smear, eraser, plate)          x
(stir, spoon, plate)            x    
(empty, plate, box)             x   
(smear, soap, plate)            x tiny
(skim off, bowl, pan)           x


candidate:
(brush, brush, bowl)            done
(brush, brush, box)             done
(dust, brush, bowl)             done (soso)
(dust, brush, pan)              done 
(dust, roller, bowl)            done
(empty, bowl, bowl)             done
(empty, bowl, plate)            done (soso)
(empty, cup, teapot)            done
(empty, teapot, plate)          done
(empty, teapot, teapot)         done
(pour in some, bowl, bowl)      done
(pour in some, cup, cup)        done (soso)
(put out, bowl, bowl)           done
(put out, bowl, pan)            done
(put out, bowl, plate)          done
(smear, glue gun, box)          done (soso)
(scrape off, knife, bowl)       done (soso)
