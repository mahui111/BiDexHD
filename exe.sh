export HYDRA_FULL_ERROR=1

cd rl_policy
# python taco_dataset/TACOdataset.py --mode make_dataset --triplet "(stir, spatula, pan)"

mode=$1
case $mode in
  train)
    # Train
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=10000 headless=True algo=ppo exp_name=init_video-grasp_reward-1.2
    ;;
  evaluate)
    # Evaluate
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 test=True checkpoint="runs/ppo_2024-07-25_22-11-15_s42/model_42500.pt"
    ;;
  debug)
    # Debug
    python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 #debug=True
    ;;
  *)
    echo "Usage: $0 {train|evaluate|debug|visualize}"
    exit 1
    ;;
esac