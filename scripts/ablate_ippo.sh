cd rl_policy

rewfunc=$1
device_list=(3 4 6 7)
triplet_list=(
    "'(dust, brush, pan)'" 
    "'(smear, glue gun, plate)'" 
    "'(empty, teapot, teapot)'"
    "'(put out, bowl, plate)'" 
)

for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    cuda_device=${device_list[$i]}
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=20000 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 headless=True rewfunc=$rewfunc &
done