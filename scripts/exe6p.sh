cd rl_policy
triplet_list=("'(brush, brush, bowl)'")

# Loop through each index in the triplet_list array
for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    CUDA_VISIBLE_DEVICES=$i python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=10000 triplet="$triplet" exp_name=ema0.1+ol2 headless=True &
done