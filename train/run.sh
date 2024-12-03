!/bin/bash
str="madry_loss" #ce_loss madry_loss"
# str2="trades_loss alp_loss"
# str3="clp_loss mart_loss mma_loss"
dataset="cifar10"
echo $str
# Loop over the words in the string
for loss in $str
do
   python -m torch.distributed.run  --standalone --nproc_per_node=gpu distributed_train.py --loss $loss --dataset $dataset
   # CUDA_VISIBLE_DEVICES=1 python train.py --loss $loss --dataset $dataset
   python test.py --loss $loss --dataset $dataset --aattack_version standard
done