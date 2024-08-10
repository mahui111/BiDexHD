cd rl_policy
triplet=$1
task_id=$2

# debug
# python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id exp_name=debug 

# visualize
# python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
# python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=8192 triplet="$triplet" task_id=$task_id exp_name=observe2 headless=True #checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(empty, bowl, bowl)/task0/ippo_Bbvdex_reward_linmin_exp_fall1/model_400.pt\
# '"

# evaluate
python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'\
/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(empty, bowl, bowl)/task0/observe2/model_8000.pt\
'"

