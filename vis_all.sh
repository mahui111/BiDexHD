# export CUDA_LAUNCH_BLOCKING=1

cd rl_policy
triplet=$1
cleaned_triplet=${triplet//[\'\"]/}
if [ ! -f "taco_dataset/sampled_data/$cleaned_triplet.json" ]; then
  echo "Making dataset for $triplet"
  python taco_dataset/TACOdataset.py --mode make_dataset --triplet "$cleaned_triplet" --num_max 6
fi

echo "Visualizing $cleaned_triplet"
for i in {0..6}
do
  echo "Task ID: $i"
  python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 test=True mode=visualize triplet="'$cleaned_triplet'" task_id=$i
done