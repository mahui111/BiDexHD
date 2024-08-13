cd rl_policy
triplet=$1
cleaned_triplet=${triplet//[\'\"]/}
if [ ! -f "taco_dataset/task_data/$cleaned_triplet.json" ]; then
    python taco_dataset/TACOdataset.py --mode make_mano_dataset --triplet "$triplet" 
fi

# debug
# python main.py task=BiLeapHandGraspV3 train=LeapHandGraspMultiPPO algo=ippo num_envs=10 triplet="$triplet" exp_name=debug 

# visualize
# python main.py task=BiLeapHandGraspV3 train=LeapHandGraspMultiPPO algo=ippo num_envs=1 triplet="$triplet" test=True mode=visualize

# train
python main.py task=BiLeapHandGraspV3 train=LeapHandGraspMultiPPO algo=ippo num_envs=5500 triplet="$triplet" exp_name=ema0.1+b2 headless=True

# evaluate
# python main.py task=BiLeapHandGraspV3 train=LeapHandGraspMultiPPO algo=ippo num_envs=1 triplet="$triplet" test=True checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs/(empty, bowl, bowl)/task9/ema0.1/model_14000.pt\
# '"

