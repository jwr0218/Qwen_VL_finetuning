# Qwen 2.5 VL Finetuning Guide

## 환경 설정

### Docker 환경
```bash
# Docker Image
huggingface/transformers-pytorch-gpu

# 실행 명령
sh train_shell.sh
```

## Training Arguments 설정

### 최적화 설정
```python
optim="adafactor"
learning_rate=self.config.learning_rate
lr_scheduler_type="constant"
warmup_ratio=self.config.warmup_ratio
max_grad_norm=self.config.max_grad_norm
```

### 정밀도 설정
```python
bf16=True
fp16=False       
bf16_full_eval=True
```

### 로깅 및 평가
```python
logging_steps=self.config.logging_steps
eval_steps=self.config.eval_steps
eval_strategy="steps"
save_strategy="steps"
save_steps=self.config.save_steps
metric_for_best_model="eval_loss"
greater_is_better=False
load_best_model_at_end=True
```

### 모델 저장
```python
save_safetensors=True
```

### 분산 학습 (FSDP)
```python
fsdp="full_shard auto_wrap"
fsdp_config={
    "min_num_params": 0,
    "xla": False,
    "xla_fsdp_grad_ckpt": False,
    "backward_prefetch": "backward_pre",
    "forward_prefetch": False,
    "limit_all_gathers": True,
    "use_orig_params": False,
    "cpu_offload": False,
    "sync_module_states": True,  # 중요: 모듈 상태 동기화
}
ddp_find_unused_parameters=False  # FSDP 사용 시 False 권장
```

### 데이터 처리
```python
dataloader_num_workers=1
dataset_text_field=""
dataset_kwargs={"skip_prepare_dataset": True}
```

### 메모리 최적화
```python
gradient_checkpointing_kwargs={"use_reentrant": False}
```

### 실험 추적
```python
report_to=["wandb"]
run_name=self.config.wandb_project_name
```

### 기타 설정
```python
local_rank=-1
```

## 주요 특징

### 🚀 성능 최적화
- **FSDP (Fully Sharded Data Parallel)**: 대용량 모델의 메모리 효율적 분산 학습
- **BFloat16 정밀도**: 메모리 사용량 감소 및 학습 속도 향상
- **Gradient Checkpointing**: 메모리 사용량 추가 감소

### 🧠 메모리 관리
- **Adafactor 옵티마이저**: 메모리 효율적인 적응형 학습률
- **CPU Offload 비활성화**: GPU 메모리 최대 활용
- **모듈 상태 동기화**: 분산 학습 시 일관성 보장

### 📊 모니터링
- **WandB 통합**: 실시간 학습 진행 상황 추적
- **단계별 평가**: 정기적인 모델 성능 검증
- **최적 모델 저장**: 검증 손실 기반 최적 체크포인트 보존

### 🛡️ 안정성
- **SafeTensors 형식**: 안전한 모델 저장
- **상수 학습률**: 안정적인 학습 진행
- **Gradient Clipping**: 학습 불안정성 방지

## 사용 방법

1. **환경 준비**
   ```bash
   # Docker 컨테이너 실행
   docker run --gpus all -it huggingface/transformers-pytorch-gpu
   ```

2. **학습 실행**
   ```bash
   sh train_shell.sh
   ```

3. **모니터링**
   - WandB 대시보드에서 실시간 학습 상황 확인
   - 로그 파일을 통한 세부 진행 상황 추적

## 주의사항

⚠️ **메모리 관리**
- 모델 크기에 따라 배치 크기 조절 필요
- GPU 메모리 부족 시 `cpu_offload=True` 고려

⚠️ **데이터 타입 일관성**
- 모든 텐서가 동일한 dtype(bfloat16) 유지 필요
- 체크포인트 재시작 시 dtype 검증 필수

⚠️ **분산 학습**
- FSDP 사용 시 `ddp_find_unused_parameters=False` 필수
- 모듈 상태 동기화 활성화 권장
