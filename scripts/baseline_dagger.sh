cd rl_policy
device_id=${1:-0}
triplet=${2:-""}
objectOffset=${3:-0.2}

# train
CUDA_VISIBLE_DEVICES=$device_id python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger expertModel=PPO num_envs=10000 triplet=$triplet objectOffset=$objectOffset exp_name=dagger_avhuber headless=True

# evaluate
# CUDA_VISIBLE_DEVICES=$device_id python main.py task=BiLeapHandGraspM3Dagger train=LeapHandGraspM3Dagger algo=m3dagger num_envs=100 triplet=$triplet objectOffset=$objectOffset test=True headless=True checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/exp2/empty/distill5/dagger_avhuber/dagger_6500.pt\
# '"