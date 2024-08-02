cd rl_policy
triplet=$1
task_id=$2
# visualize
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=5120 triplet="$triplet" task_id=$task_id exp_name=ippo_Bbvdex_reward_0.3ho_exp headless=True

# evaluate
# python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'runs/(hit, hammer, box)/task7/ippo_Bbvdex_reward_0.3ho_exp/model_3000.pt'"

