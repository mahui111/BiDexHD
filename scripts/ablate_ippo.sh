cd rl_policy

rewfunc=$1  # ab_stage1, ab_funcgrasp, ab_bonus
device_list=(0 1 2 3 4 5 6 7)
# triplet_list=(
#     "'(dust, brush, pan)'" 
#     "'(smear, glue gun, plate)'" 
#     "'(empty, teapot, teapot)'"
#     "'(put out, bowl, plate)'"  
# )
triplet_list=(
    "'(dust, brush, bowl)'"         
    "'(empty, bowl, bowl)'"         
    "'(empty, bowl, plate)'"        
    "'(empty, cup, plate)'"         
    "'(empty, teapot, plate)'"      
    "'(pour in some, teapot, cup)'" 
    "'(put out, bowl, bowl)'"       
    "'(skim off, bowl, plate)'"     
)
for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    cuda_device=${device_list[$i]}
    CUDA_VISIBLE_DEVICES=$cuda_device python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=20000 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 headless=True rewfunc=$rewfunc &
done