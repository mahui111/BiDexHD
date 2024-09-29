cd rl_policy

triplet_list=(
    "'(dust, brush, bowl)'"
    "'(dust, brush, pan)'"
    "'(empty, bowl, bowl)'"
    "'(empty, bowl, plate)'"
    "'(empty, cup, plate)'"
    "'(empty, teapot, plate)'"
    "'(empty, teapot, teapot)'"
    "'(pour in some, cup, cup)'"
    "'(pour in some, cup, plate)'"
    "'(pour in some, cup, teapot)'"
    "'(pour in some, teapot, bowl)'"
    "'(pour in some, teapot, cup)'"
    "'(put out, bowl, bowl)'"
    "'(put out, bowl, plate)'"
    "'(skim off, bowl, plate)'"
    "'(smear, glue gun, plate)'"
)
for i in "${!triplet_list[@]}"; do
# triplet=$1
# cleaned_triplet=${triplet//[\'\"]/}
# if [ ! -f "taco_dataset/sampled_data/$cleaned_triplet.json" ]; then
  triplet=${triplet_list[$i]}
  cleaned_triplet=${triplet//[\'\"]/}
  echo "Making dataset for $cleaned_triplet"
  python taco_dataset/TACOdataset.py --mode make_dataset --triplet "$cleaned_triplet" --num_max 200
# fi
done

# echo "Visualizing $cleaned_triplet"
# for i in {0..20}
# do
#   echo "Task ID: $i"
#   python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 test=True mode=visualize triplet="'$cleaned_triplet'" task_id=$i
# done
