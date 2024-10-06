GPU=0
LAMBDA=${1}  ## {0.0004, 0.0008, 0.0032, 0.01 0.045}
ARCH=base  ###  【base, dwt, dwt_gate】
LOSS=mse
DATA_SET=UC_real_  #_Urban  UC_real_  DOTA_real_
MODEL_PATH=checkpoint/ELIC_arch_${ARCH}/${LOSS}_${LAMBDA}
TEST_PATH=results/ELIC_arch_${ARCH}/${DATA_SET}_${LOSS}_${LAMBDA}
mkdir -p ${TEST_PATH}
export CUDA_VISIBLE_DEVICES=$GPU
cp ${MODEL_PATH}/checkpoint_best_loss.pth.tar ./
python3 updata.py   checkpoint_best_loss.pth.tar -n  mid_test  --arch ${ARCH}
rm ./checkpoint_best_loss.pth.tar
python3 Inference.py \
      --output_path ${TEST_PATH} \
      --arch ${ARCH} \
      -p  mid_test.pth.tar \
      --dataset /media/xjtuei-h/11/LJH/image_compression/ELIC_Grad_Guide/results/ELIC_arch_Test_GT_GEN/mse_0.0008 \
       --patch 64


#/media/xjtuei-h/11/LJH/image_compression/ELIC_Grad_Guide/results/ELIC_arch_Test_UC_GT_GEN/mse_0.0008_png
#/media/xjtuei-h/11/LJH/image_compression/ELIC_Grad_Guide/results/ELIC_arch_Test_GT_GEN/mse_0.0008
#/media/xjtuei-h/11/LJH/image_compression/RS_datasets/LOVEDA/datasets_crop/ELIC_arch_Test_GT_LOVEDA/mse_0.0008
#/media/xjtuei-h/11/LJH/image_compression/RS_datasets/LOVEDA/datasets_crop/ELIC_arch_Test_GT_Urban_256
#/media/xjtuei-h/11/LJH/image_compression/RS_datasets/LOVEDA/datasets_crop/ELIC_arch_Test_GT_LOVEDA/mse_0.0008  \

# --dataset ./../RS_datasets/DOTA_V1.5/test_sub99_512/   \
      #  -p ./pre_weights/ELIC_0008_ft_3980_Plateau.pth.tar \
#python3 -m updata.py checkpoint -n ./checkpoint/ELIC_arch_base/mse_0.045/checkpoint_best_loss_159.pth.tar

# --dataset ./../RS_datasets/DIOR/obb_dataset/gen_dataset/tmp_DIOR_test_RIO  \