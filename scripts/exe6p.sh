cd rl_policy
triplet_list=(
    "'(brush, brush, bowl)'"
    "'(dust, brush, bowl)'"
    "'(dust, brush, pan)'"
    "'(dust, roller, bowl)'"
    "'(empty, bowl, bowl)'"
    "'(empty, bowl, plate)'"
    "'(empty, cup, teapot)'"
    "'(empty, teapot, plate)'"
)

# triplet_list=(
#     "'(empty, teapot, teapot)'"
#     "'(put out, bowl, bowl)'"
#     "'(put out, bowl, pan)'"
#     "'(put out, bowl, plate)'"
#     "'(pour in some, bowl, bowl)'"
#     "'(pour in some, cup, cup)'"
#     "'(scrape off, knife, bowl)'"
#     "'(smear, glue gun, box)'"
# )


# Loop through each index in the triplet_list array
for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    CUDA_VISIBLE_DEVICES=$i python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=10000 triplet="$triplet" exp_name=ema0.1+ol2 headless=True &
done