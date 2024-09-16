cd rl_policy

triplet_list=(
    "'(empty, bowl, bowl)'"               
    "'(empty, bowl, plate)'"               
    "'(empty, cup, plate)'"               
    "'(empty, teapot, plate)'"               
    "'(empty, teapot, teapot)'"               
    "'(pour in some, cup, cup)'"               
    "'(pour in some, cup, plate)'"               
    "'(pour in some, cup, teapot)'"               
)

# triplet_list=(
#     "'(pour in some, teapot, bowl)'"
#     "'(pour in some, teapot, cup)'" 
#     "'(dust, brush, bowl)'"
#     "'(dust, brush, pan)'"   
#     "'(put out, bowl, bowl)'"    
#     "'(put out, bowl, plate)'"     
#     "'(smear, glue gun, plate)'"       
#     "'(skim off, bowl, plate)'"        
# )

for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    CUDA_VISIBLE_DEVICES=$i python main.py task=BiLeapHandGraspV6 train=LeapHandGraspBaselinePPO algo=ppo num_envs=20000 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 headless=True &
done