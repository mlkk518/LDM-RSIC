GPU=0
LAMBDA=0.01  ## 【0.045  0.01  0.0032】
ARCH=$1  ###  【base, dwt, dwt_gate】
GATE_Weight=0.5 ###  【base, dwt, dwt_gate】
LOSS=mse
MODEL_PATH=checkpoint/ELIC_arch_${ARCH}/${LOSS}_${LAMBDA}
mkdir -p $MODEL_PATH
export CUDA_VISIBLE_DEVICES=$GPU
python3 train.py --N 192 --M 320 \
        --epochs 3000 \
        -lr 1e-5 \
        --num-workers 8 \
        --arch ${ARCH} \
        --gate_weight ${GATE_Weight} \
        --lambda ${LAMBDA} \
        --batch-size 8 \
        --test-batch-size 8 \
        --aux-learning-rate 1e-4 \
        --dataset ./../RS_datasets/DOTA_v1.5_UC/ \
        --savepath ${MODEL_PATH}/ \
        --checkpoint  ${MODEL_PATH}/checkpoint_best_loss.pth.tar

#        --dataset ../RS_datasets/DOTA_V1.5/ \


# lambda |  Link        论文中  比其小一个数量级                                                                                      |
#| ----|---------------------------------------------------------------------------------------------------|
#| 0.45 | [0.45](https://drive.google.com/file/d/1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN/view?usp=sharing)    |
#| 0.15 | [0.15](https://drive.google.com/file/d/1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL/view?usp=sharing)    |
#| 0.032 | [0.032](https://drive.google.com/file/d/1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc/view?usp=sharing)    |
#| 0.016 | [0.0016](https://drive.google.com/file/d/1MWlYAmpHbWlGtG7MBBTPEew800grY5yC/view?usp=sharing)     |
#| 0.008| [0.008](https://drive.google.com/file/d/1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL/view?usp=sharing) |
#| 0.004 | [0.004](https://drive.google.com/file/d/1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q/view?usp=sharing)    |