cd rl_policy
mode=$1

case $mode in
  train)
    # Train
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo
    ;;
  evaluate)
    # Evaluate
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 checkpoint="runs/ppo_2024-07-25_14-41-27_s42/model_10000.pt"
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