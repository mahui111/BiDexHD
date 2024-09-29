# export CUDA_LAUNCH_BLOCKING=1
cd rl_policy

Kfuturestep=${1:-5}

verbs=(
    "dust"
    "empty" 
    "pourinsome"
    "putout"
    "smear"
    "skimoff"
)

for i in "${!verbs[@]}"; do
    cudadevice=$(($i))
    verb=${verbs[$i]}
    CUDA_VISIBLE_DEVICES=$cudadevice python main.py task=BiLeapHandGraspM3Dagger train=$verb triplet=$verb algo=m3dagger num_envs=10000 objectOffset=0.2 exp_name=dagger_avhuber test=True headless=True Kfuturestep=$Kfuturestep &
done

