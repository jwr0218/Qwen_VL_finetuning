import torch
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from qwen_vl_utils import process_vision_info

import torch
import torch.nn as nn
from transformers import Trainer



import torch
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class WeightedVLDataCollator:
    """
    기존 collate_fn 로직을 기반으로, 'text_parts'에 가중치를 부여하는 Data Collator.
    'text_parts' 키가 JSON에 포함되어 있다고 가정합니다.
    """
    processor: Any  # QwenVLProcessor
    text_weight: float = 5.0  # 텍스트에 부여할 가중치 (예: 5.0)
    bbox_weight: float = 1.0  # 그 외 (BBox, 특수문자 등) 가중치 (예: 1.0)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # 1. 기존 collate_fn 로직 (이미지 및 텍스트 준비)
        # print(features)
        # exit()
        texts = [
            self.processor.apply_chat_template(example, tokenize=False) 
            for example in features
        ]
        image_inputs = [process_vision_info(example)[0] for example in features]

        # print('check===========================')
        # print(texts)
        # print('check===========================')
        # exit()
        # 이미지 로드 (경로가 유효하지 않을 경우를 대비한 예외 처리 추가)
        # images = []
        # for sample in features:
        #     try:
        #         images.append(Image.open(sample['image_path']).convert('RGB'))
        #     except FileNotFoundError:
        #         print(f"Warning: Image file not found {sample['image_path']}. Using blank image.")
        #         images.append(Image.new('RGB', (224, 224), (255, 255, 255))) # 임시 이미지

        # 2. Processor를 사용한 일괄 처리
        inputs = self.processor(
            text=texts, 
            images=image_inputs, 
            return_tensors='pt', 
            padding=True
        )

        # 3. 'labels' 생성 (기존 collate_fn 로직과 동일)
        labels = inputs.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # 4. (NEW) 'loss_weights' 텐서 생성
        # 기본 가중치(bbox_weight)로 초기화
        loss_weights = torch.full_like(labels, self.bbox_weight, dtype=torch.float)

        for i, sample in enumerate(features): # 'sample'은 챗 리스트 (예: [{'role':'user',...}, ...])
            
            # --- 5a. 쿼리 부분 마스킹 (Qwen-VL 템플릿 기반으로 수정) ---
            
            # Qwen-VL 템플릿은 '<|im_start|>assistant\n' 같은 특수 토큰으로 턴을 구분합니다.
            # 이 토큰의 *마지막* 등장을 찾아 그 *이후*를 'answer'로 간주합니다.
            
            # 표준 Qwen 템플릿의 assistant 시작 구분자
            assistant_prefix_tokens = self.processor.tokenizer(
                "\n<|im_start|>assistant\n", 
                add_special_tokens=False
            ).input_ids

            full_input_ids_sample = inputs.input_ids[i]
            answer_start_index = -1
            
            # 토큰 시퀀스에서 마지막 assistant prefix 위치 검색
            for k in range(len(full_input_ids_sample) - len(assistant_prefix_tokens) + 1):
                window = full_input_ids_sample[k : k + len(assistant_prefix_tokens)]
                if torch.equal(window, torch.tensor(assistant_prefix_tokens, device=window.device)):
                    answer_start_index = k + len(assistant_prefix_tokens) # 실제 답변 텍스트 시작 위치

            if answer_start_index == -1:
                # Fallback: `\n`이 없는 경우 (대화의 첫 턴이 assistant일 경우, 드물지만)
                assistant_prefix_tokens_alt = self.processor.tokenizer(
                    "<|im_start|>assistant\n", add_special_tokens=False).input_ids
                for k in range(len(full_input_ids_sample) - len(assistant_prefix_tokens_alt) + 1):
                     window = full_input_ids_sample[k : k + len(assistant_prefix_tokens_alt)]
                     if torch.equal(window, torch.tensor(assistant_prefix_tokens_alt, device=window.device)):
                        answer_start_index = k + len(assistant_prefix_tokens_alt)
            
            if answer_start_index == -1:
                print(f"Warning: [Sample {i}] Could not find assistant prefix. Masking may be incorrect.")
                continue # 이 샘플은 마스킹/가중치 적용 스킵

            # 쿼리 부분(답변 시작 전)은 loss 계산에서 제외
            labels[i, :answer_start_index] = -100
            loss_weights[i, :answer_start_index] = 0.0

            # --- 5b. 'text_parts'에 가중치 적용 (접근 방식 수정) ---
            
            # (중요한 가정) 
            # 'text_parts' 리스트가 Dataset에서 'assistant' 메시지 딕셔너리 안에
            # {'role': 'assistant', 'content': [...], 'text_parts': ['...']}
            # 와 같이 포함되어 전달된다고 가정합니다.
            try:
                text_parts = sample[-1].get("text_parts", []) # sample[-1]이 assistant 딕셔너리
            except (IndexError, AttributeError, TypeError):
                text_parts = [] # 샘플이 비어있거나, 딕셔너리 구조가 아닌 경우
            
            if not text_parts:
                continue

            for text_part in text_parts:
                text_part_ids = self.processor.tokenizer(text_part, add_special_tokens=False).input_ids
                text_part_len = len(text_part_ids)
                
                if text_part_len == 0:
                    continue

                # (수정됨) 전체 시퀀스가 아닌, '답변' 영역 내에서만 텍스트 검색
                answer_region_ids = full_input_ids_sample[answer_start_index:]
                
                for k in range(len(answer_region_ids) - text_part_len + 1):
                    window = answer_region_ids[k : k + text_part_len]
                    
                    if torch.equal(window, torch.tensor(text_part_ids, device=window.device)):
                        
                        # 찾으면, *전체 시퀀스* 기준의 인덱스로 변환
                        global_start_index = answer_start_index + k
                        global_end_index = global_start_index + text_part_len
                        
                        # 해당 위치의 가중치를 text_weight로 변경
                        loss_weights[i, global_start_index : global_end_index] = self.text_weight
                        
                        # (가정: 텍스트가 답변 내에서 한 번만 등장)
                        break 
                            
        # 6. 최종 패딩 마스킹 (기존 코드 유지)
        # labels가 -100인 모든 부분 (쿼리 + 패딩)의 가중치를 0.0으로 확실하게 설정
        loss_weights[labels == -100] = 0.0

        # 7. 최종 배치 반환 (기존 코드 유지)
        inputs['labels'] = labels
        inputs['loss_weights'] = loss_weights  # 가중치 텐서 추가

        return inputs
    




class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Data Collator로부터 'loss_weights'를 받아 가중 손실을 계산합니다.
        """
        
        # 1. 입력에서 labels와 loss_weights를 분리
        #    이 키들은 Data Collator에서 생성한 키와 일치해야 합니다.
        labels = inputs.pop("labels", None)
        loss_weights = inputs.pop("loss_weights", None)

        # 2. 모델을 포워딩하여 logits를 얻음 (labels를 넘기지 않음)
        outputs = model(**inputs)
        logits = outputs.logits

        # 3. 가중 손실 계산
        loss = None
        if labels is not None and loss_weights is not None:
            # Causal LM의 표준 loss 계산 (shift)
            # logits: (batch_size, seq_len, vocab_size)
            # labels: (batch_size, seq_len)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()

            # Loss 함수를 'reduction='none''으로 초기화 (토큰별 loss 계산)
            loss_fct = nn.CrossEntropyLoss(reduction='none')

            # Logits와 labels의 차원을 맞춤
            vocab_size = shift_logits.size(-1)
            
            # (batch_size * (seq_len - 1))
            loss_per_token = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
            # (batch_size, seq_len - 1)
            loss_per_token = loss_per_token.view(shift_labels.shape)

            # 4. 가중치 적용
            weighted_loss = loss_per_token * shift_weights

            # 5. 최종 Loss 계산 (가중치의 합으로 정규화)
            # 패딩(-100)이 아닌 유효한 토큰들의 가중치 합으로 나눔
            valid_weights_sum = shift_weights.sum()
            
            if valid_weights_sum > 0:
                loss = weighted_loss.sum() / valid_weights_sum
            else:
                # 유효한 토큰이 없는 예외적인 경우
                loss = weighted_loss.sum()

        elif labels is not None:
            # loss_weights가 없으면 (예: eval) 기본 loss 사용
            loss = outputs.loss if outputs.loss is not None else super().compute_loss(model, {**inputs, "labels": labels})

        return (loss, outputs) if return_outputs else loss
    
