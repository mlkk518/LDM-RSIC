LAMBDA=${1}
GPU=${2} #,1,2,3 #,1,2,3,4
export CUDA_VISIBLE_DEVICES=${GPU}
#python3 DiffIR/train.py -opt options/train_DiffIR_JPEG2K_S2_${LAMBDA}bpp.yml
python3 DiffIR/train.py -opt options/train_DiffIR_mlkkS2_${LAMBDA}.yml