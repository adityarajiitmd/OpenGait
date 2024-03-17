# pytorch framework
# this is for testing purpose script
# CUDA_VISIBLE_DEVICES specifies which GPUs to use for the experiment. The number of GPUs varies depending on the experiment and dataset.
# The commands use torch.distributed.launch which indicates distributed training or testing across multiple GPUs.
# --cfgs argument points to the configuration file (YAML) for the specific experiment.
# When running distributed training/testing across multiple machines (nodes), processes (workers) need to communicate and coordinate their actions.
# master_port defines a specific port on which a process, called the master process, listens for communication from other worker processes.
# nproc_per_node This argument specifies the number of processes (workers) to launch per node when using torch.distributed.launch. Each process will handle training/testing on a single GPU.


# # **************** For CASIA-B ****************
# # Baseline
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/baseline/baseline.yaml --phase test

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitset/gaitset.yaml --phase test

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitpart/gaitpart.yaml --phase test

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgl/gaitgl.yaml --phase test

# # GLN 
# # Phase 1
# CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port 12345  --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase1.yaml --phase test
# # Phase 2
# CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gln/gln_phase2.yaml --phase test


# # **************** For OUMVLP ****************
# # Baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/baseline/baseline_OUMVLP.yaml --phase test

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitset/gaitset_OUMVLP.yaml --phase test

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_OUMVLP.yaml --phase test

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_OUMVLP.yaml --phase test
