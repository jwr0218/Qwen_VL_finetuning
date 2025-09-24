import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import requests
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
from transformers import (
    AutoProcessor,
)
from torch.cuda.amp import GradScaler

from accelerate import Accelerator
import wandb

from toonVLM.qwen_multi_head import Multihead_Toonspace
from toonVLM.train_utils_multihead import VLMTrainer




from torch.cuda.amp import GradScaler, autocast

from toonVLM.TrainingConfig import TrainingConfig




def main():
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"
    
    print(f"디바이스: {device} | 모델: {model_name}\n")
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    
    # 모델 불러오기 & Optimizer 불러오기 
    model = Multihead_Toonspace(model_name).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    
    # 학습 실행
    trainer = VLMTrainer(config,model,processor)
    
    trainer.train()
if __name__ == "__main__":
    main()

