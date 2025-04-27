export DATASET_NAME=synapse-b
export CUDA_DEVICE_NUM=7
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name kl-f4
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name kl-f8
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name kl-f16
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name kl-f32
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name vq-f4
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name vq-f4-noattn
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name vq-f8
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name vq-f8-n256
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUM python -W ignore -u scripts/evalVAE.py \
    --dataset $DATASET_NAME \
    --vae_name vq-f16
