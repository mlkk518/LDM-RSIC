LAMBDA=${1}
GPU=${2} #,1,2,3 #,1,2,3,4
export CUDA_VISIBLE_DEVICES=${GPU}
#python3  DiffIR/test.py -opt options/test_DiffIRS_JPEG2K_${LAMBDA}bpp.yml
python3  DiffIR/test.py -opt options/test_DiffIRS2_${LAMBDA}.yml



## Conda Env  compress