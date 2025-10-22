"""
차트 질문 답변을 위한 Vision Language Model 파인튜닝 스크립트 (리팩토링 버전)

이 모듈은 차트 분석 작업을 위한 Qwen2.5-VL 모델의 파인튜닝을 구현합니다.
전체 파인튜닝(Full Fine-tuning)을 사용하여 차트 이미지와 
관련 질문 및 답변을 처리합니다.

작성자: 정원렬
Year: 2025
Month: 8
Day: 27
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from accelerate import Accelerator, PartialState
from toonVLM.TrainingConfig import TrainingConfig
from tqdm.auto import tqdm


class VLMTrainer:
    """Vision Language Model 파인튜닝을 위한 메인 클래스"""
    
    def __init__(self, config =  None, model = None, processor = None):
        self.config = config
        self.setup_logging()
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        # Accelerator 초기화
        self.accelerator = Accelerator()
        
    def setup_logging(self) -> None:
        """로깅 시스템 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def clear_memory(self) -> None:
        """안전한 GPU 메모리 정리"""
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                if torch.cuda.memory_allocated() > 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                
                self.logger.info(f"GPU allocated memory: {allocated_gb:.2f} GB")
                self.logger.info(f"GPU reserved memory: {reserved_gb:.2f} GB")
                
            except RuntimeError as e:
                self.logger.warning(f"CUDA 동기화 중 오류 발생: {e}")
                self.logger.warning("GPU 메모리 정리를 건너뜁니다.")
        else:
            self.logger.warning("CUDA를 사용할 수 없습니다.")
        
        gc.collect()
    
    def format_data(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        비전-언어 모델 학습을 위한 입력 데이터 포맷팅
        
        Args:
            sample: 원시 데이터 샘플
            
        Returns:
            포맷된 대화 형태의 데이터
        """
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ]
    
    def load_and_split_dataset(self) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """
        데이터셋 로드 및 분할
        
        Returns:
            train_dataset, eval_dataset, test_dataset
        """
        try:
            if not Path(self.config.data_path).exists():
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.config.data_path}")
            
            dataset = load_dataset('json', data_files=self.config.data_path)
            dataset['train'] = dataset["train"].shuffle(seed=42)
            total_size = len(dataset["train"])
            
            # 데이터셋 분할 인덱스 계산
            train_size = int(self.config.train_ratio * total_size)
            eval_size = int(self.config.eval_ratio * total_size)
            
            # 데이터셋 분할
            train_dataset = dataset["train"].select(range(0, train_size))
            eval_dataset = dataset["train"].select(range(train_size, train_size + eval_size))
            test_dataset = dataset["train"].select(range(train_size + eval_size, train_size + eval_size * 2))
            
            # # 데이터 포맷팅
            # train_formatted = [self.format_data(sample) for sample in train_dataset]
            # eval_formatted = [self.format_data(sample) for sample in eval_dataset]
            # test_formatted = [self.format_data(sample) for sample in test_dataset]
            
            # self.logger.info(f"데이터셋 로드 완료:")
            # self.logger.info(f"  - 학습 데이터: {len(train_formatted)}개")
            # self.logger.info(f"  - 검증 데이터: {len(eval_formatted)}개") 
            # self.logger.info(f"  - 테스트 데이터: {len(test_formatted)}개")
            
            # 샘플 데이터 출력
            # if train_formatted and self.accelerator.is_main_process:
                # self.logger.info(f"샘플 데이터:\n{train_formatted[0]}")
            
            return train_dataset , eval_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"데이터셋 로드 중 오류 발생: {e}")
            raise
    
    def load_model_and_processor(self) -> None:
        """모델 및 프로세서 로드"""
        try:
            self.logger.info(f"모델 로드 시작: {self.config.model_id}")
            
            if self.model is None:

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model_id,
                    # device_map="auto",  # `accelerate`가 device_map을 자동 관리
                    torch_dtype=torch.bfloat16
                )
            else:
                print('prameter model이 업로드 되었습니다.\n')
            if self.processor is None:

                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_id,
                    min_pixels=self.config.min_pixels,
                    max_pixels=self.config.max_pixels
                )

            else:
                print('prameter processor가 업로드 되었습니다.\n')
            
            self.logger.info("모델 및 프로세서 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def create_collate_fn(self):
        """간단한 배치 생성 - 모델이 내부에서 처리"""
        def collate_fn(examples: List[Dict]) -> Dict[str, any]:
            from PIL import Image
            
            # 이미지 로드
            images = []
            for example in examples:
                if isinstance(example['image'], str):
                    image = Image.open(example['image']).convert('RGB')
                else:
                    image = example['image']
                images.append(image)
            
            # OCR과 DESCRIPTION 텍스트 추출
            ocr_texts = [example.get('OCR', '') for example in examples]
            desc_texts = [example.get('DESCRIPTION', '') for example in examples]
            pre_desc_texts = [example.get('previous_desc', '') for example in examples]
            ocr_labels = self.tokenizer(ocr_texts[0], return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids#.to(device)
            desc_labels = self.tokenizer(desc_texts[0], return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids#.to(device)
            pre_desc_labels = self.tokenizer(pre_desc_texts[0], return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids#.to(device)
            # 배치 딕셔너리 생성
            batch = {
                "image": image,           # List[PIL.Image]
                'previous_desc' : pre_desc_labels , 
                "ocr": ocr_labels,         # List[str]  
                "desc": desc_labels         # List[str]
            }
            
            return batch
        
        return collate_fn


    def train(self) -> None:
        """메인 학습 파이프라인 실행"""
        try:
            self.logger.info("VLM 파인튜닝 시작")
            
            # 메모리 정리
            self.clear_memory()
            
            # 데이터 로드
            train_dataset, eval_dataset, test_dataset_list = self.load_and_split_dataset()
            
            # 모델 및 옵티마이저 로드
            self.load_model_and_processor()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            
            # 데이터 콜레이터 생성
            collate_fn = self.create_collate_fn()
            
            # 데이터 로더 생성
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            
            # `accelerate`로 모델, 옵티마이저, 데이터로더 준비
            self.model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, eval_dataloader
            )
            
            self.logger.info("학습 시작")
            
            # 커스텀 학습 루프
            for epoch in range(self.config.num_train_epochs):
                self.model.train()
                print(f'EPOCH : \t{epoch}')
                for step, batch in enumerate(tqdm(train_dataloader)):
                    # 그래디언트 초기화를 먼저 수행
                    optimizer.zero_grad()
                    
                    # autocast 사용 (GPU에서만 활성화)
                    # print(batch.keys())
                    with torch.cuda.amp.autocast(enabled=(self.accelerator.device.type == 'cuda'), dtype=torch.bfloat16):
                        try:

                            total_loss = self.model(
                                image=batch['image'],
                                previous_desc = batch['previous_desc'],
                                ocr_labels=batch['ocr'],
                                desc_labels=batch['desc'],
                                processor=self.processor,
                                tokenizer=self.tokenizer
                            )
                        except torch.OutOfMemoryError as e: 
                                print(e)
                                print('\n\n 에러 발생  - 컨티뉴 합니다.')
                                continue

                    if step % self.config.logging_steps == 0 :
                        self.logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {total_loss.item():.4f}")
                    
                    # NaN 체크 후 역전파 수행
                    if not torch.isnan(total_loss):
                        # accelerator를 사용한 역전파
                        self.accelerator.backward(total_loss)
                        
                        # 그래디언트 클리핑
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # 옵티마이저 스텝 실행
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        self.logger.warning(f"손실이 NaN이므로 Epoch {epoch+1}, Step {step+1}을 건너뜁니다.")
                        continue
                # exit()
                # 평가 루프
                self.model.eval()
                eval_loss = 0
                eval_steps = 0
                
                for batch in eval_dataloader:
                    with torch.no_grad():
                        # autocast를 평가에도 적용
                        with torch.cuda.amp.autocast(enabled=(self.accelerator.device.type == 'cuda'), dtype=torch.bfloat16):
                            
                            outputs = self.model(**batch)
                            
                            if not torch.isnan(outputs.loss):
                                eval_loss += outputs.loss.item()
                                eval_steps += 1
                
                if eval_steps > 0:
                    avg_eval_loss = eval_loss / eval_steps
                    self.logger.info(f"Epoch {epoch+1}, Avg Eval Loss: {avg_eval_loss:.4f}")
                else:
                    self.logger.warning(f"Epoch {epoch+1}: 모든 평가 배치에서 NaN 손실 발생")
            
            # 메인 프로세스에서만 모델 저장
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self.logger.info(f"모델 저장: {self.config.output_dir}")
                unwrapped_model.save_pretrained(self.config.output_dir)
                self.processor.save_pretrained(self.config.output_dir)
            
            self.logger.info("학습 완료")
            
        except RuntimeError as e:
            self.logger.error(f"학습 중 오류 발생: {e}")
            raise
        finally:
            self.clear_memory()


def main():
    """메인 실행 함수"""
    try:
        # 설정 로드
        config = TrainingConfig()
        
        # 학습 실행
        trainer = VLMTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()