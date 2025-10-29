import copy
import logging
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

# --- 로거 설정 (선택 사항) ---
back_ext_logger = logging.getLogger("BackExtension")
back_ext_logger.setLevel(logging.INFO)
if not back_ext_logger.hasHandlers():
    back_ext_logger.addHandler(logging.StreamHandler())
# ------------------------------


# --- 파라미터 동결/해제 헬퍼 함수 ---

def _freeze_layer_parameters(layer: nn.Module):
    """헬퍼 함수: 레이어의 모든 파라미터를 동결합니다 (학습 X)."""
    for param in layer.parameters():
        param.requires_grad = False

def _unfreeze_layer_parameters(layer: nn.Module):
    """헬퍼 함수: 레이어의 모든 파라미터를 학습 가능하게 합니다 (학습 O)."""
    for param in layer.parameters():
        param.requires_grad = True

# -----------------------------------


def apply_back_extension_with_freeze(
    original_model: Qwen2_5_VLForConditionalGeneration, 
    target_layers: int
) -> Qwen2_5_VLForConditionalGeneration:
    """
    Qwen-VL 모델 객체에 Back Extension (Zero-Init Deepening)을 
    직접 적용합니다. 이 함수는 모델을 "in-memory"에서 수정합니다.

    - 원본 레이어: 동결 (Freeze)
    - 추가된 레이어: 학습 가능 (Unfreeze) 및 Zero-Init
    - (수정됨) config 파일은 수정하지 않습니다.

    Args:
        original_model (Qwen2_5_VLForConditionalGeneration): 
            .from_pretrained()로 로드된 원본 모델 객체.
        target_layers (int): 
            목표로 하는 총 레이어 수.

    Returns:
        Qwen2_5_VLForConditionalGeneration: 
            레이어가 추가되고 수정된 모델 객체.
    """
    
    # 1. 원본 레이어 수 확인 (config에서 읽기만 함)
    try:
        original_layers = original_model.config.num_hidden_layers
    except AttributeError:
        back_ext_logger.error("오류: 'config.llm_config.num_hidden_layers'를 찾을 수 없습니다.")
        back_ext_logger.error("입력 모델이 Qwen2_5_VL 타입이 맞는지 확인하세요.")
        raise
        
    if original_layers >= target_layers:
        back_ext_logger.warning(
            f"목표 레이어 수({target_layers})가 원본 레이어 수({original_layers})보다 "
            f"크지 않습니다. 원본 모델을 그대로 반환합니다."
        )
        return original_model

    back_ext_logger.info(
        f"[Back Extension] 적용 중... "
        f"원본: {original_layers} 레이어 -> 목표: {target_layers} 레이어"
    )

    # 2. 복제 간격(split) 계산
    num_layers_to_add = target_layers - original_layers
    split = original_layers // num_layers_to_add
    
    if split == 0:
        back_ext_logger.warning(
            f"추가할 레이어 수({num_layers_to_add})가 원본({original_layers})보다 "
            f"많습니다. 'split'을 1로 강제합니다."
        )
        split = 1
        
    back_ext_logger.info(
        f"원본 레이어 {split}개마다 1개의 레이어를 복제합니다. (총 {num_layers_to_add}개 추가)"
    )

    # 3. LLM 레이어 경로 접근
    try:
        original_llm_layers = original_model.layers
    except AttributeError:
        back_ext_logger.error("오류: 'model.language_model.model.layers' 경로를 찾을 수 없습니다.")
        back_ext_logger.error("모델 아키텍처가 다르거나 손상되었을 수 있습니다.")
        raise

    new_llm_layers = nn.ModuleList()
    
    layer_cnt = 0
    layers_added = 0

    # 4. 레이어 순회 및 복제
    for i in range(original_layers):
        original_layer = original_llm_layers[i]
        
        # 4a. 원본 레이어 동결 (Freeze)
        _freeze_layer_parameters(original_layer)
        
        new_llm_layers.append(original_layer)
        layer_cnt += 1
        
        # 4b. 복제 조건 확인
        if (i + 1) % split == 0 and layers_added < num_layers_to_add:
            back_ext_logger.info(f"  -> 레이어 {i} 복제 -> 새 레이어 {layer_cnt}로 추가")
            
            # 5. 레이어 복제
            new_layer = copy.deepcopy(original_llm_layers[i])
            
            # 6. 새 레이어만 학습 가능하도록 동결 해제
            _unfreeze_layer_parameters(new_layer)

            # 7. Zero-Initialization (가중치 0으로 초기화)
            try:
                # Qwen 2.5 아키텍처 기준
                new_layer.self_attn.o_proj.weight.data.zero_()
                if new_layer.self_attn.o_proj.bias is not None:
                    new_layer.self_attn.o_proj.bias.data.zero_()
                    
                new_layer.mlp.down_proj.weight.data.zero_()
                if new_layer.mlp.down_proj.bias is not None:
                    new_layer.mlp.down_proj.bias.data.zero_()
                    
            except AttributeError as e:
                back_ext_logger.error(f"Zero-Init 중 오류: {e}. 아키텍처가 다를 수 있습니다.")
                raise e
            
            new_llm_layers.append(new_layer)
            layer_cnt += 1
            layers_added += 1

    # 8. 모델 객체에 새 레이어 리스트 적용
    back_ext_logger.info(f"LLM 레이어 리스트를 {layer_cnt}개로 교체합니다.")
    original_model.model.language_model.model.layers = new_llm_layers
    
    # 9. (수정됨) config 수정 로직 제거
    #    FSDP/Trainer가 모델 객체 자체의 구조를 보도록 함
    
    # 10. (선택사항) FSDP 등이 참조할 수 있도록 내부 속성 업데이트
    try:
        original_model.model.language_model.model.num_layers = layer_cnt
    except AttributeError:
        pass # 이 속성이 없어도 치명적이지 않음

    back_ext_logger.info(
        f"모델 Deepening 완료. 새 레이어 수: {layer_cnt}"
    )
    
    return original_model