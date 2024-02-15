export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --architecture resnet18 --model unet-cond-ca-bs512-150K --embedding_type random_synthetic_uniform_5000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --architecture resnet18 --model unet-cond-ca-bs512-150K --embedding_type random_synthetic_two_stage_5000


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --architecture resnet18 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_uniform_5000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --architecture resnet18 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_two_stage_5000



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --architecture resnet18 --model unet-cond-ca-bs512-150K-cpd50 --embedding_type random_synthetic_uniform_5000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --architecture resnet18 --model unet-cond-ca-bs512-150K-cpd50 --embedding_type random_synthetic_two_stage_5000



# ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
