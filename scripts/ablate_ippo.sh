cd rl_policy
mode=${1:-train}
rewfunc=$2  # ab_stage1, ab_funcgrasp, ab_bonus
device_list=(0 1 2 3 4 5 6 7)
triplet_list=(
  "(empty, bowl, bowl)"               
  "(empty, bowl, plate)"               
  "(empty, cup, plate)"               
  "(empty, teapot, plate)"               
  "(empty, teapot, teapot)"               
  "(pour in some, cup, cup)"               
  "(pour in some, cup, plate)"               
  "(pour in some, cup, teapot)"   
  "(pour in some, teapot, bowl)"
  "(pour in some, teapot, cup)"
  "(dust, brush, bowl)"
  "(dust, brush, pan)"   
  "(put out, bowl, bowl)"    
  "(put out, bowl, plate)" 
  "(skim off, bowl, plate)"        
  "(smear, glue gun, plate)"  
)

# training
if [ "$mode" == "train" ]; then
    for i in "${!triplet_list[@]}"; do
        triplet=${triplet_list[$i]}
        cuda_device=${device_list[$i]}
        CUDA_VISIBLE_DEVICES=$cuda_device python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=20000 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 headless=True rewfunc=$rewfunc &
    done

# evaluate
else
    triplet=$3
    train_ids=${4:-[]}
    cleaned_triplet=${triplet//[\'\"]/}
    checkpoint=$(find /home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-$rewfunc/freq3/exp2/"$cleaned_triplet"/ema0.1+ol2_oo0.2/2w/ -name 'model_*.pt')
    python main.py task=BiLeapHandGraspV6 train=LeapHandGraspMultiPPO algo=ippo num_envs=100 triplet="$triplet" objectOffset=0.2 exp_name=ema0.1+ol2_oo0.2 test=True headless=True train_ids="$train_ids" rewfunc=multippo checkpoint="'$checkpoint'"
fi