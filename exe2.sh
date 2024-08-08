cd rl_policy
triplet=$1
task_id=$2
# visualize
python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=6000 triplet="$triplet" task_id=$task_id exp_name=reimplement headless=True observationType="dofps+dofvel+ftps+lastact+palmpose" #checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(empty, bowl, bowl)/task0/ippo_Bbvdex_reward_linmin_exp_fall1/model_400.pt\
# '"

# evaluate
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(pour in some, teapot, cup)/task1/noobj/model_19500.pt\
# '"

