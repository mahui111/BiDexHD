cd rl_policy
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo triplet="'(empty, bowl, bowl)'"
# python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 debug=True checkpoint="runs/ppo_2024-07-25_10-35-26_s42/model_5500.pt"

# tensorboard --logdir rl_policy/runs/