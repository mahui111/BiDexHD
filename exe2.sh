cd rl_policy
triplet=$1
task_id=$2
cleaned_triplet=${triplet//[\'\"]/}
if [ ! -f "taco_dataset/task_data/$cleaned_triplet.json" ]; then
    python taco_dataset/TACOdataset.py --mode make_mano_dataset --triplet "$triplet" #--dataset_dir "/home/ps/Desktop/zbh/robot/BVDex/assets/TACOdataset/overall/" --mano_model_path /home/ps/Desktop/zbh/robot/BVDex/rl_policy/taco_dataset/manopth/mano/models
fi

# debug
# python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id exp_name=debug 

# visualize
# python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True mode=visualize

# train
# python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=6144 triplet="$triplet" task_id=$task_id exp_name=observe2HO headless=True

# evaluate
python main.py task=BiLeapHandGraspV2 train=LeapHandGraspPPO algo=ippo num_envs=1 triplet="$triplet" task_id=$task_id test=True checkpoint="'\
/home/ps/Desktop/zbh/robot/BVDex/rl_policy/runs/(put out, spatula, pan)/task0/observe2/model_500.pt\
'"

