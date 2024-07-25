cd rl_policy
mode=$1

case $mode in
  train)
    # Train
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo checkpoint="runs/ppo_2024-07-25_21-00-45_s42/model_1500.pt"
    ;;
  evaluate)
    # Evaluate
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=100 test=True checkpoint="runs/ppo_2024-07-25_21-00-45_s42/model_1500.pt"
    ;;
  debug)
    # Debug
    python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 #debug=True
    ;;
  visualize)
    # Visualize
    tensorboard --logdir rl_policy/runs/
    ;;
  *)
    echo "Usage: $0 {train|evaluate|debug|visualize}"
    exit 1
    ;;
esac