triplet=$1
cleaned_triplet=${triplet//[\'\"]/}
test=${2:-False}

DPATH="/mnt/hpfs/baairl/zbh/BVDex/rl_policy/runs-multippo/$cleaned_triplet/ema0.1+ol2"
bash rl_policy/runs-multippo/down.sh "$DPATH"

if [ "$test" = "True" ]; then
    cd rl_policy
    ckptdir="runs-multippo/114-cgpos-clip/$cleaned_triplet/ema0.1+ol2/"
    ckptfile=$(find "$ckptdir" -type f -name "*.pt" | sort | tail -n 1)
    python main.py task=BiLeapHandGraspV5 train=LeapHandGraspMultiPPO algo=ippo num_envs=100 triplet="$triplet" test=True checkpoint="'$ckptfile'"
    cd ..
fi

# echo -e "\n$cleaned_triplet\t\t\t(done)" >> README.md
