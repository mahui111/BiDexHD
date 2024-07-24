cd rl_policy
python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True
# python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=1 checkpoint="runs/BiLeapHandGrasp_2024-07-23_22-29-03/model_10000.pt"

# tensorboard --logdir rl_policy/runs/