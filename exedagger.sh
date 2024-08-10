cd rl_policy
triplet=$1
task_id=$2

# debug
# python -m pdb main.py task=BiLeapHandGraspPCD train=LeapHandGraspDagger algo=dagger num_envs=10 triplet="$triplet" task_id=$task_id exp_name=debug_dagger 

# visualize
# python main.py task=BiLeapHandGraspPCD train=LeapHandGraspDagger algo=dagger num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
# python main.py task=BiLeapHandGraspPCD train=LeapHandGraspDagger algo=dagger num_envs=500 triplet="$triplet" task_id=$task_id exp_name=dagger_avhuberclip headless=True 

# evaluate
python main.py task=BiLeapHandGraspPCD train=LeapHandGraspDagger algo=dagger num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'\
/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(empty, bowl, bowl)/task0/dagger_avhuberclip/dagger_2000.pt\
'"

