cd rl_policy
task_path="/home/zbh/Desktop/zbh/robot/TACO-Instructions/dataset/overall/Hand_Poses"

# Check if a task list is provided
if [ -n "$1" ]; then
    # Use the provided task list
    task_list=("$@")
else
    # Default to all directories in the task path and extract them as a list
    mapfile -t task_list < <(find "$task_path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \;)
fi
echo "Number of tasks: ${#task_list[@]}"
echo "tasks: ${task_list[@]}"
for triplet in "${task_list[@]}"; do
    python taco_dataset/TACOdataset.py --mode make_mano_dataset --triplet "'$triplet'" --num_max 200
done