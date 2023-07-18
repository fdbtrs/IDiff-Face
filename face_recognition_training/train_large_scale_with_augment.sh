export OMP_NUM_THREADS=2

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_jonas.py --architecture resnet50 --model unet-cond-ca-bs512-150K-cpd25 --embedding_type random_synthetic_uniform_15000  --width 10000 --depth 50 --augment True



# ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
