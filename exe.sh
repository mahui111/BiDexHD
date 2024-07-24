cd rl_policy
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ippo
# python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=1 checkpoint="runs/BiLeapHandGrasp_2024-07-24_22-57-58_s42/model_2000.pt"

# tensorboard --logdir rl_policy/runs/