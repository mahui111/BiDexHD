# export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
triplet=$1
train_ids=${2:-[]} # Default to empty array if not provided
objoffset=${3:-0.2} # Default to 0.1 if not provided
cleaned_triplet=${triplet//[\'\"]/}
if [ ! -f "taco_dataset/task_data/$cleaned_triplet.json" ]; then
    python taco_dataset/TACOdataset.py --mode make_mano_dataset --triplet "$triplet"  --num_max 200
fi

# debug
# python -m pdb main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=110 train_ids="$train_ids" objectOffset=$objoffset triplet="$triplet" exp_name=debug 

# visualize
# python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=1 train_ids="$train_ids" objectOffset=$objoffset triplet="$triplet" test=True mode=visualize

# train
# CUDA_VISIBLE_DEVICES=0 python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=20000 train_ids="$train_ids" objectOffset=$objoffset triplet="$triplet" exp_name=ema0.1+ol2 headless=True

# evaluate
python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=100 triplet="$triplet" train_ids="$train_ids" objectOffset=$objoffset headless=True test=True checkpoint="'\
/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-multippo/freq3/exp2/(brush, brush, bowl)/ema0.1+ol2_sel[1,2,3,5,9,10,11]/model_20000.pt\
'"

