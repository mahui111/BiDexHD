from glob import glob
import os.path as osp
import numpy as np
from natsort import natsorted

triplets = \
'''
(brush, brush, bowl)        
(brush, brush, box)         
(dust, brush, bowl)         
(dust, brush, pan)          
(dust, roller, bowl)        
(empty, bowl, bowl)         
(empty, bowl, plate)        
(empty, cup, teapot)        
(empty, teapot, plate)      
(empty, teapot, teapot)     
(pour in some, bowl, bowl)  
(pour in some, cup, cup)    
(put out, bowl, bowl)       
(put out, bowl, pan)        
(put out, bowl, plate)      
(smear, glue gun, box)      
(scrape off, knife, bowl)
'''   

triplets = [x.strip() for x in triplets.split('\n') if x]
cnt = 0
for triplet in triplets:
    expert_path_list = glob(f'/home/zbh/Desktop/zbh/robot/BVDex/rl_policy/runs-multippo/ready/{triplet}/ema0.1+ol2/*.pt')
    expert_path = natsorted(expert_path_list)[-1]
    cnt += len(expert_path_list)
    # if len(expert_path_list) > 1:
    #     print(f'choose from following:',expert_path_list)
    if len(expert_path_list) == 1:
        print(expert_path)
print(f'triplets: {len(triplets)}, expert_paths: {cnt}')

expertpaths = []
for path in glob('/mnt/hpfs/baairl/zbh/BVDex/rl_policy/runs-baselineppo/freq3/exp2/*'):
    expert_path_list = glob(osp.join(path, '*/*/*'))
    expert_path = natsorted(expert_path_list)[-1]
    expertpaths.append(expert_path)
for each in sorted(expertpaths):
    print(each)