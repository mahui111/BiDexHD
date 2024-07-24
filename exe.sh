cd rl_policy
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ippo
# python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ippo num_envs=1 checkpoint="runs/ippo_2024-07-25_00-54-37_s42/model_500.pt"

# tensorboard --logdir rl_policy/runs/