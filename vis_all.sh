# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1
cd rl_policy

# triplet_list=("(brush, brush, plate)" "(pour in some, teapot, teapot)" "(hit, hammer, toy)" "(pour in some, teapot, cup)" "(smear, eraser, plate)")
# for triplet in "${triplet_list[@]}"; do
#     python taco_dataset/TACOdataset.py --mode make_dataset --triplet "$triplet"
# done

for i in {0..30}
do
  echo "Task ID: $i"
  python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 test=True mode=visualize task_id=$i triplet="'(pour in some, teapot, cup)'"
done
