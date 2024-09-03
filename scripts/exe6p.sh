cd rl_policy
# triplet_list=(
#     "'(brush, brush, bowl)'"
#     "'(dust, brush, bowl)'"
#     "'(dust, brush, pan)'"
#     "'(dust, roller, bowl)'"
#     "'(empty, bowl, bowl)'"
#     "'(empty, bowl, plate)'"
#     "'(empty, cup, teapot)'"
#     "'(empty, teapot, plate)'"
# )

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

# triplet_list=(
#     "'(brush, brush, teapot)'"
#     "'(dust, brush, cup)'"
#     "'(empty, cup, plate)'"
#     "'(empty, teapot, cup)'"
#     "'(pour in some, teapot, cup)'"
#     "'(pour in some, teapot, bowl)'"
#     "'(pour in some, cup, teapot)'"
#     "'(skim off, bowl, plate)'"
# )

triplet_list=(
    "'(brush, brush, plate)'"
    "'(dust, roller, box)'"
    "'(empty, bowl, cup)'"
    "'(put in, bowl, pan)'"
    "'(put in, bowl, plate)'"
    "'(scrape off, knife, box)'"
    "'(smear, glue gun, plate)'"
    "'(smear, eraser, box)'"
)


# Loop through each index in the triplet_list array
for i in "${!triplet_list[@]}"; do
    triplet=${triplet_list[$i]}
    CUDA_VISIBLE_DEVICES=$i python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=20000 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 headless=True &
done