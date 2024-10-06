LAMBDA=${1}
GPU=${2} #,1,2,3 #,1,2,3,4
export CUDA_VISIBLE_DEVICES=${GPU}
#python DiffIR/train.py -opt options/ablation/train_DiffIR_JPEG2KS1_${LAMBDA}bpp.yml
python DiffIR/train.py -opt options/train_DiffIR_mlkkS1_${LAMBDA}.yml



#CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4393 DiffIR/train.py -opt options/train_DiffIRS1_${LAMBDA}.yml --launcher pytorch