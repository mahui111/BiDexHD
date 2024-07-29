# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1
cd rl_policy
for i in {2,3,7}
do
  echo "Task ID: $i"
  python main.py task=BiLeapHandGraspV1 train=LeapHandGraspPPO algo=ppo num_envs=1 test=True mode=visualize task_id=$i triplet="'(hit, hammer, toy)'"
done
