


# Set NCCL options for distributed training
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0

# Run training script with torchrun for distributed GPU training
# echo "Starting distributed training..."
# accelerate launch --multi_gpu --num_processes=2 train/qwen_vision_train.py
accelerate launch --multi_gpu --num_processes=2 train/custom_loss_qwen_vision_train.py
# accelerate launch --multi_gpu --num_processes=2 train/train.py
# torchrun --nproc_per_node=2 qwen_vision_test_all.py
# echo "Training completed."
