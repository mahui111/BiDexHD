# export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
triplet=$1
cleaned_triplet=${triplet//[\'\"]/}
if [ ! -f "taco_dataset/task_data/$cleaned_triplet.json" ]; then
    python taco_dataset/TACOdataset.py --mode make_mano_dataset --triplet "$triplet"  --num_max 200
fi

# debug
# python main.py task=BiLeapHandGraspV5 train=LeapHandGraspMultiPPO algo=ippo num_envs=110 triplet="$triplet" exp_name=debug 

# visualize
# python main.py task=BiLeapHandGraspV5 train=LeapHandGraspMultiPPO algo=ippo num_envs=1 triplet="$triplet" test=True mode=visualize

# train
CUDA_VISIBLE_DEVICES=4 python main.py task=BiLeapHandGraspV5 train=LeapHandGraspMultiPPO algo=ippo num_envs=10000 triplet="$triplet" exp_name=ema0.1+ol2 headless=True

# evaluate
# python main.py task=BiLeapHandGraspV5 train=LeapHandGraspMultiPPO algo=ippo num_envs=100 triplet="$triplet" test=True checkpoint="'\
# /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-multippo/(empty, bowl, bowl)/task1/ema0.1/model_500.pt\
# '"

