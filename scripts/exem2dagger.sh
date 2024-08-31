export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
triplet=$1
task_id=$2

# debug
# python -m pdb main.py task=BiLeapHandGraspM2Dagger train=LeapHandGraspM2Dagger algo=m2dagger num_envs=550 exp_name=debug_dagger # checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/distill4/dagger_avhuber/dagger_2000.pt\
# '"


# visualize
# python main.py task=BiLeapHandGraspM2Dagger train=LeapHandGraspM2Dagger algo=m2dagger num_envs=1 test=True mode=visualize

# train
# python main.py task=BiLeapHandGraspM2Dagger train=LeapHandGraspM2Dagger algo=m2dagger num_envs=1500 exp_name=dagger_avhuber headless=True #checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/distill1/dagger_avhuber/dagger_27000.pt\
# '"

# evaluate
python main.py task=BiLeapHandGraspM2Dagger train=LeapHandGraspM2Dagger algo=m2dagger num_envs=500 test=True headless=True checkpoint="'\
/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-dagger/verb/distill3/dagger_ahuber2_putout/dagger_3000.pt\
'"


