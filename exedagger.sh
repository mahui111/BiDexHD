cd rl_policy
triplet=$1
task_id=$2

# debug
python -m pdb main.py task=BiLeapHandGraspMultiDagger train=LeapHandGraspDagger algo=dagger num_envs=9 exp_name=debug_dagger 

# visualize
# python main.py task=BiLeapHandGraspMultiDagger train=LeapHandGraspDagger algo=dagger num_envs=1 test=True mode=visualize

# train
# python main.py task=BiLeapHandGraspMultiDagger train=LeapHandGraspDagger algo=dagger num_envs=500 exp_name=dagger_avhuber headless=True 

# evaluate
# python main.py task=BiLeapHandGraspMultiDagger train=LeapHandGraspDagger algo=dagger num_envs=1 test=True checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(pour in some, teapot, cup)/task1/dagger_avhuber/dagger_500.pt\
# '"

