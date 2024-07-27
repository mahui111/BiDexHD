export HYDRA_FULL_ERROR=1

cd rl_policy
# python taco_dataset/TACOdataset.py --mode make_dataset --triplet "(stir, spatula, pan)"

mode=$1
case $mode in
  train)
    # Train
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO num_envs=8192 headless=True algo=ppo exp_name=bvdex_reward-height 
    ;;
  evaluate)
    # Evaluate
    python main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 test=True checkpoint="'runs/(stir, spatula, pan)/bvdex_grasp_reward-object/model_15000.pt'"
    ;;
  debug)
    # Debug
    python -m pdb main.py task=BiLeapHandGrasp train=LeapHandGraspPPO algo=ppo num_envs=1 mode=visualize  #debug=True
    ;;
  *)
    echo "Usage: $0 {train|evaluate|debug|visualize}"
    exit 1
    ;;
esac