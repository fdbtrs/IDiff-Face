export OMP_NUM_THREADS=2

# large-scale experiments with CA-CPD25 (Uniform)
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_uniform_15000 --width 5000 --depth 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_uniform_15000 --width 5000 --depth 32
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_uniform_15000 --width 10000 --depth 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_uniform_15000 --width 10000 --depth 50

# large-scale experiments with CA-CPD25 (Two-Stage)
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_two_stage_15000 --width 5000 --depth 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_two_stage_15000 --width 5000 --depth 32
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_two_stage_15000 --width 10000 --depth 16
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_two_stage_15000 --width 10000 --depth 50

# large-scale experiments with CA-CPD50 (Two-Stage)
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd50 --embedding_type random_synthetic_two_stage_15000  --width 5000 --depth 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd50 --embedding_type random_synthetic_two_stage_15000  --width 5000 --depth 32
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd50 --embedding_type random_synthetic_two_stage_15000  --width 10000 --depth 16
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
#--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd50 --embedding_type random_synthetic_two_stage_15000  --width 10000 --depth 50
