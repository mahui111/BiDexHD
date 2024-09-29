cd rl_policy
mode=${1:-train}
verbs=(
    "dust"
    # "empty" 
    # "pourinsome"
    # "putout"
    # "smear"
    # "skimoff"
)

# train
if [ "$mode" == "train" ]; then
    for i in "${!verbs[@]}"; do
        cudadevice=$(($i))
        verb=${verbs[$i]}
        CUDA_VISIBLE_DEVICES=$cudadevice python main.py task=BiLeapHandGraspBC train=$verb triplet=$verb algo=bc num_envs=10000 objectOffset=0.2 exp_name=dagger_avhuber headless=True &
    done

# evaluate
else
    checkpoint=$2
    verb=$(basename $(dirname $(dirname $checkpoint)))
    python main.py task=BiLeapHandGraspBC train=$verb triplet=$verb algo=bc num_envs=100 objectOffset=0.2 test=True checkpoint="$checkpoint"
fi
