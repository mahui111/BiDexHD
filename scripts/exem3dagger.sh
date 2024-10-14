export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
device_id=${1:-0}
triplet=${2:-""}
objectOffset=${3:-0.2}

# debug
# python -m pdb main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=550 exp_name=debug_dagger # checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/distill4/dagger_avhuber/dagger_2000.pt\
# '"


# visualize
# python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=1 test=True mode=visualize

# train
# CUDA_VISIBLE_DEVICES=$device_id python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=5000 triplet=$triplet objectOffset=$objectOffset exp_name=dagger_avhuber headless=True #checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/distill1/dagger_avhuber/dagger_27000.pt\
# '"n

# evaluate
CUDA_VISIBLE_DEVICES=$device_id python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=1 triplet=$triplet objectOffset=$objectOffset test=True headless=False checkpoint="'\
/home/zbh/Desktop/zbh/robot/BiDexHD/rl_policy/runs-dagger/real_pour in some/dagger_avhuber/dagger_8500.pt\
'"


