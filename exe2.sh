cd rl_policy
triplet=$1
task_id=$2
# visualize
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=6144 triplet="$triplet" task_id=$task_id exp_name=ippo_Bbvdex_reward_min_2+0.2lift_exp headless=True isMinReward=1

# evaluate
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'runs/(stir, spoon, pan)/task9/ippo_Bbvdex_reward_min_exp/model_14500.pt'"

