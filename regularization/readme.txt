# to run distributed train
python -m torch.distributed.run  --standalone --nproc_per_node=gpu distributed_train.py