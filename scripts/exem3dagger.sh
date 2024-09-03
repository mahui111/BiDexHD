export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
triplet=$1
objectOffset=${2:-0.1}

# debug
# python -m pdb main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=550 exp_name=debug_dagger # checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/distill4/dagger_avhuber/dagger_2000.pt\
# '"


# visualize
# python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=1 test=True mode=visualize

# train
# CUDA_VISIBLE_DEVICES=4 python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=5000 objectOffset=$objectOffset exp_name=dagger_avhuber headless=True #checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/distill1/dagger_avhuber/dagger_27000.pt\
# '"

# evaluate
python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=100 test=True checkpoint="'\
/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/exp/distill2-0.9/dagger_avhuber/dagger_9000.pt\
'"


