cd rl_policy

triplet_list=(
    "(pour in some, teapot, cup)" 
    "(dust, roller, box)"
)

echo "Triplet List: ${triplet_list[@]}"

# Enumerate over triplet_list
for i in "${!triplet_list[@]}"; do
    cudadevice=$(($i))
    
    triplet=${triplet_list[$i]}
    cleaned_triplet=${triplet//[\'\"]/}
    
    if [ ! -f "taco_dataset/task_data/$cleaned_triplet.json" ]; then
        echo "Making dataset for $triplet"
        python taco_dataset/TACOdataset.py --mode make_mano_dataset --triplet "$triplet" --num_max 200 || { echo "Dataset creation failed for $triplet"; exit 1; }
    fi
    
    # Train with enumerated index
    CUDA_VISIBLE_DEVICES=$cudadevice python main.py task=BiLeapHandGraspV3 train=LeapHandGraspMultiPPO algo=ippo num_envs=15000 triplet="'$cleaned_triplet'" exp_name=ema0.1 headless=True &

    # Uncomment to evaluate
    # python main.py task=BiLeapHandGraspV3 train=LeapHandGraspMultiPPO algo=ippo num_envs=100 triplet="$triplet" test=True checkpoint="'\
    # /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-multippo/(empty, bowl, bowl)/task1/ema0.1/model_500.pt\
    # '"
done
wait
