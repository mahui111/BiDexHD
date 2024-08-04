cd rl_policy
triplet=$1
task_id=$2
# visualize
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=6500 triplet="$triplet" task_id=$task_id exp_name=ippo_Bbvdex_reward_linmin_exp headless=True

# evaluate
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'runs/(hit, hammer, box)/task5/ippo_Bbvdex_reward_linmin_exp/model_500.pt'"

