# export CUDA_LAUNCH_BLOCKING=1

cd rl_policy
triplet_list=("(put out, bowl, bowl)" "(pour in some, bowl, bowl)" "(put in, bowl, bowl)" "(skim off, bowl, bowl)" "(empty, bowl, bowl)")
for triplet in "${triplet_list[@]}"; do
  cleaned_triplet=${triplet//[\'\"]/}
  if [ ! -f "taco_dataset/sampled_data/$cleaned_triplet.json" ]; then
    echo "Making dataset for $triplet"
    python taco_dataset/TACOdataset.py --mode make_dataset --triplet "$triplet"
  fi
done

for triplet in "${triplet_list[@]}"; do
  echo "Visualizing $triplet"
  for i in {1..20}
  do
    echo "Task ID: $i"
    python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 test=True mode=visualize triplet="'$triplet'" task_id=$i
  done
done