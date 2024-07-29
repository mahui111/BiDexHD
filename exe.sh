# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
# python taco_dataset/TACOdataset.py --mode make_dataset --triplet "(stir, spatula, pan)"

mode=$1
case $mode in
  train)
    # Train
    python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO num_envs=8192 headless=True algo=ippo exp_name=ippo_grasp_reward
    ;;
  evaluate)
    # Evaluate
    python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ippo num_envs=1 test=True checkpoint="'runs/(stir, spoon, pan)/task4/ippo_grasp_reward/model_500.pt'"
    ;;
  visualize)
    # Visualize
    python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 test=True mode=visualize 
    ;;
  debug)
    # Debug
    python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 #debug=True
    ;;
  *)
    echo "Usage: $0 {train|evaluate|debug|visualize}"
    exit 1
    ;;
esac