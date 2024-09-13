cd rl_policy

# triplet_list=(
#     "'(brush, brush, teapot)'"
#     "'(dust, brush, bowl)'"
#     "'(dust, brush, cup)'"
#     "'(dust, roller, bowl)'"
#     "'(empty, bowl, bowl)'"
#     "'(empty, bowl, cup)'"
#     "'(empty, bowl, plate)'"
#     "'(empty, cup, plate)'"
# )

triplet_list=(
    "'(empty, cup, teapot)'"
    "'(empty, teapot, cup)'"
    "'(empty, teapot, plate)'"
    "'(empty, teapot, teapot)'"
    "'(pour in some, bowl, bowl)'"
    "'(pour in some, cup, cup)'"
    "'(pour in some, cup, teapot)'"
    "'(pour in some, teapot, bowl)'"
)

# triplet_list=(
#     "'(pour in some, teapot, cup)'"
#     "'(put in, bowl, plate)'"
#     "'(put out, bowl, bowl)'"
#     "'(put out, bowl, plate)'"
#     "'(scrape off, knife, bowl)'"
#     "'(skim off, bowl, bowl)'"
#     "'(skim off, bowl, plate)'"
#     "'(smear, glue gun, plate)'"
# )


# Loop through each index in the triplet_list array
for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    CUDA_VISIBLE_DEVICES=$i python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=50000 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 headless=True &
done