import torch 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm 
# Re-importing libraries after the environment reset
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction
from rouge_score import rouge_scorer
# import metrics as deplot_metric
from torch.utils.data import Dataset
import json
import numpy as np 
from PIL import Image

from transformers import AutoProcessor, BitsAndBytesConfig


class WebtoonDataset(Dataset):
    """웹툰 VLM 학습용 데이터셋"""
    
    def __init__(self, data_list):
        """
        Args:
            data_list (list): 각 요소가 다음 중 하나:
                - 원시 데이터: {"image_path": ..., "query": ..., "answer": ...}
                - 포맷된 데이터: [{"role": "system", ...}, {"role": "user", ...}, ...]
        """
        self.data = data_list
        print(f"데이터셋 초기화 완료: {len(data_list)}개 샘플")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def format_data(sample, task_type="general"):
    """
    웹툰 비전-언어 모델 학습을 위한 입력 데이터를 포맷팅합니다.
    
    Args:
        sample (dict): 원시 데이터 샘플로 다음을 포함:
            - image_path (str) 또는 image (PIL Image): 웹툰 이미지
            - query (str): 웹툰에 대한 사용자 질문
            - answer/label (str/dict): 예상 답변/응답
        task_type (str): Task 유형
    
    Returns:
        list: 역할(system, user, assistant)과 이미지 및 텍스트 구성 요소를 포함한 포맷된 대화
    """
    
    # Task별 시스템 메시지
    system_messages = {
        "scene_description": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰 이미지를 분석하여 장면의 상황, 맥락, 서사적 흐름을 정확하게 설명하세요.",
        
        "text_detection": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰 이미지에서 모든 텍스트 요소를 정확히 인식하고 위치 정보와 함께 추출하세요.",
        
        "terminology": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰에서 사용된 특별한 용어들을 식별하고 그 의미를 설명하세요.",
        
        "character_analysis": "당신은 웹툰 이미지 분석 전문가입니다. 웹툰 캐릭터들을 분석하여 등장인물 정보와 감정 상태를 파악하세요.",
        
        "general": "당신은 웹툰 이미지 분석 전문가입니다. 성인 웹툰 이미지를 분석하여 장면별로 효과음, 말풍선, 서사적 맥락을 정확히 추출하고, JSON 형식으로 구조화된 결과를 제공합니다. 모든 텍스트 요소(대사, 효과음, 나레이션)를 한국어로 추출하고, 캐릭터 관계와 상황 맥락을 세밀히 분석하며, 오해석을 최소화하십시오"
    }
    
    # 이미지 로드 처리
    image = None
    if "image" in sample:
        image = sample["image"]
    elif "image_path" in sample:
        image = Image.open(sample["image_path"]).convert("RGB")
    else:
        raise ValueError("이미지 정보가 없습니다. 'image' 또는 'image_path' 키가 필요합니다.")
    
    # 답변 처리 (label 또는 answer 키 모두 지원)
    answer = sample.get("answer", sample.get("label", ""))
    if isinstance(answer, dict):
        # dict 형태의 답변을 JSON 문자열로 변환
        answer = json.dumps(answer, ensure_ascii=False, indent=2)
    
    system_message = system_messages.get(task_type, system_messages["general"])
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

def clear_memory():
    """
    안전한 GPU 메모리 정리 함수
    
    CUDA 초기화 상태를 확인하고 안전하게 메모리를 정리합니다.
    오류 발생 시에도 프로그램이 중단되지 않도록 예외 처리를 포함합니다.
    """
    import gc
    import torch
    
    # Python 가비지 컬렉션 먼저 실행
    gc.collect()
    
    # CUDA가 사용 가능하고 초기화되었는지 확인
    if torch.cuda.is_available():
        try:
            # GPU 메모리가 할당되어 있는지 확인
            if torch.cuda.memory_allocated() > 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
        except RuntimeError as e:
            print(f"CUDA 동기화 중 오류 발생: {e}")
            print("GPU 메모리 정리를 건너뜁니다.")
    else:
        print("CUDA를 사용할 수 없습니다.")
    
    # 마지막으로 한 번 더 가비지 컬렉션
    gc.collect()

def create_collate_fn(processor, task_type="general"):
    """
    특정 processor와 task_type을 위한 collate_fn을 생성합니다.
    
    Args:
        processor: 모델의 프로세서 (tokenizer + image processor)
        task_type (str): Task 유형
    
    Returns:
        function: Trainer에서 사용할 수 있는 collate_fn
    """
    
    def collate_fn(examples):
        """
        웹툰 비전-언어 모델 학습을 위한 사용자 정의 데이터 콜레이션 함수입니다.
        
        대화 예제 배치를 다음과 같이 처리합니다:
        1. 대화를 포맷하기 위해 채팅 템플릿 적용
        2. 이미지에서 비전 정보 추출 및 처리
        3. 텍스트 토큰화 및 이미지 입력 준비
        4. 학습을 위한 적절한 라벨 생성 (패딩 토큰 마스킹)
        
        Args:
            examples (list): 원시 데이터 샘플 목록 또는 포맷된 대화 예제 목록
        
        Returns:
            dict: 다음을 포함한 배치 딕셔너리:
                - input_ids: 토큰화된 입력 시퀀스
                - attention_mask: 패딩을 위한 어텐션 마스크
                - pixel_values: 처리된 이미지 텐서
                - labels: 학습을 위한 타겟 라벨 (패딩 토큰은 -100)
        """
        
        # Task 타입 자동 감지 (수동 설정이 우선)
        current_task_type = task_type
        if task_type == "auto" and examples:
            # 첫 번째 예제가 dict인지 list인지 확인
            first_example = examples[0]
            if isinstance(first_example, dict):
                query = first_example.get("query", "").lower()
                if "장면" in query or "상황" in query or "묘사" in query:
                    current_task_type = "scene_description"
                elif "텍스트" in query or "말풍선" in query or "인식" in query:
                    current_task_type = "text_detection"
                elif "용어" in query or "의미" in query:
                    current_task_type = "terminology"
                elif "캐릭터" in query or "등장인물" in query or "감정" in query:
                    current_task_type = "character_analysis"
                else:
                    current_task_type = "general"
        
        # 예제가 이미 포맷된 대화 형태인지 확인
        if examples and isinstance(examples[0], list):
            # 이미 format_data()로 포맷된 경우
            formatted_examples = examples
        else:
            # 원시 데이터인 경우 포맷팅 필요
            formatted_examples = [
                format_data(example, current_task_type) for example in examples
            ]
        
        # 채팅 템플릿 적용하여 텍스트 생성
        texts = [
            processor.apply_chat_template(formatted_example, tokenize=False) 
            for formatted_example in formatted_examples
        ]
        
        # 이미지 추출 (각 예제의 user 메시지에서 이미지 추출)
        image_inputs = [process_vision_info(formatted_example)[0] for formatted_example in formatted_examples]
        
        # 프로세서로 배치 처리
        batch = processor(
            text=texts, 
            images=image_inputs, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        
        # 라벨 생성 (패딩 토큰은 -100으로 마스킹)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch
    
    return collate_fn

def load_webtoon_data(json_path):
    """
    웹툰 JSON 데이터를 로드하고 처리합니다.
    
    Args:
        json_path (str): JSON 파일 경로
    
    Returns:
        list: 처리된 데이터 리스트
    """
    print(f"JSON 파일 로딩 중: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"로드된 데이터 수: {len(data)}")
        print("데이터 구조 확인...")
        
        if data:
            print("첫 번째 데이터 키들:", list(data[0].keys()))
        
        return data
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        raise
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {e}")
        raise


checkpoint_path = "/workspace/Toonspace_VLM/ex_models/qwen25-7b-Webtoon_Analysis"

# 4 bit 

# quant_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_enable_fp32_cpu_offload=False,

# )

# 8 bit

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint_path,
    # quantization_config=quant_config,   # 주석 처리하면 full-precision
    device_map="auto",
    torch_dtype='auto'
)

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     checkpoint_path, torch_dtype=torch.bfloat16, device_map="auto"
# )

# print(model)
# exit()
processor = AutoProcessor.from_pretrained(
     checkpoint_path,
     padding_side = "left",
min_pixels=256*28*28,
max_pixels=960*28*28
 )
# print(model)
# print(processor.tokenizer.padding_side)
# exit()


model.eval()

print(checkpoint_path)



def get_measure(refer, predict):
    # BLEU score 계산 함수
    def calculate_bleu(reference, prediction):
        smooth = SmoothingFunction().method1
        return sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)

    # BLEU-RT score 계산 함수: 기존 BLEU 점수에 Brevity Penalty 조정을 추가
    def calculate_bleu_rt(reference, prediction):
        candidate = prediction.split()
        ref_tokens = reference.split()
        candidate_len = len(candidate)
        reference_len = len(ref_tokens)
        
        # 기본 BLEU score 계산 (smoothing 적용)
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], candidate, smoothing_function=smooth)
        
        # 단순화한 Brevity Penalty 적용:
        # 후보 길이가 기준보다 짧을 경우 패널티를 부여
        if candidate_len > reference_len:
            bp = 1.0
        else:
            bp = np.exp(1 - (reference_len / candidate_len))
        
        return bleu * bp

    # ROUGE score 계산 함수
    def calculate_rouge(reference, prediction):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores

    # F1 score 계산 (단어 단위)
    def calculate_f1(reference, prediction):
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        common = set(reference_tokens) & set(prediction_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(prediction_tokens)
        recall = len(common) / len(reference_tokens)
        return 2 * (precision * recall) / (precision + recall)
    
    bleu_score = calculate_bleu(refer, predict)
    bleu_rt = calculate_bleu_rt(refer, predict)
    rouge_scores = calculate_rouge(refer, predict)
    f1 = calculate_f1(refer, predict)

    return {'bleu': bleu_score,
            'bleu-rt': bleu_rt,
            'rouge': rouge_scores['rouge1'].fmeasure,
            'f1': f1}


# nia_path = '/workspace/data_moder/GPT_TABLE.json'
# nia_path = '/workspace/data_moder/GPT_TABLE_DES_QA.json'
webtoon_path = '/workspace/Toonspace_VLM/data/json_file/webtoon_balanced_training.json' # GPT_ALL <- query가 바뀐 GPT


data_lst = {'total_path' : webtoon_path }



for name , path in data_lst.items():

    print(path)

    raw_data = load_webtoon_data(path)
    # 데이터셋 분할
    total_len = len(raw_data)
    print(f"전체 데이터 수: {total_len}")
    
    if total_len < 10:
        print("경고: 데이터가 너무 적습니다. 최소 10개 이상의 샘플이 필요합니다.")
    
    dataset_num = int(0.95 * total_len)
    eval_start = dataset_num
    eval_end = min(eval_start + 10, total_len)
    test_start = eval_end
    test_end = min(test_start + 10, total_len)
    
    train_dataset = raw_data[:dataset_num]
    eval_dataset = raw_data[eval_start:eval_end] if eval_end > eval_start else []
    test_dataset = raw_data[test_start:test_end] if test_end > test_start else []


    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(eval_dataset)}개") 
    print(f"테스트 데이터: {len(test_dataset)}개")
    
    # Task 타입 설정 (자동 감지 또는 수동 설정)
    task_type = "auto"  # "scene_description", "text_detection", "terminology", "character_analysis", "general", "auto"
    
    # 데이터 포맷팅 (선택적 - collate_fn에서도 처리 가능)
    print("데이터 포맷팅 중...")
    test_formatted = [format_data(sample, task_type) for sample in test_dataset]
    
    # Dataset 클래스로 래핑
    test_dataset = WebtoonDataset(test_formatted) if test_formatted else None

    f1_lst = []
    rouge_lst = []
    bleu_lst = []
    bleu_rt_lst = []
    RMS_precision_lst = []
    RMS_recall_lst = []
    RMS_f1_lst = []
    RNSS_lst = []
    batch_size = 8

    def evaluate_batch(batch):
        # 메시지와 입력 데이터 준비
        texts, image_inputs_list, video_inputs_list, references = [], [], [], []
        # print(len(batch))
        # print(batch[0])
        # print('=========='*10)
        # print(batch[0][0])
        # print('=========='*10)
        # print(batch[0][1])
        # print('=========='*10)
        # print(batch[0][2])
        # print('=========='*10)

        
        for data in batch:

            # print(data[2].keys())
            # print(data[2]['content'])
            references.append(data[2]['content'][0]['text'])  # 참조 데이터 저장
            # 메시지 포맷
            # print(data[1].keys())
            messages = [data[0],data[1]]

            # Text와 Vision 정보 생성
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            texts.append(text)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)

        # 배치 단위로 모델 입력 준비
        inputs = processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 생성
        generated_ids = model.generate(**inputs, max_new_tokens=5000, do_sample=True)

        # 배치로 생성된 출력 디코딩
        output_texts = []
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
            generated_ids_trimmed = out_ids[len(in_ids):]
            decoded = processor.decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_texts.append(decoded)
        


        return output_texts, references

    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        # 현재 배치

        batch = test_dataset[i : i + batch_size]
        batch_generated, batch_references = evaluate_batch(batch)
        # 메트릭 계산
        for predicted in batch_generated:
            print(predicted)
            print('=========='*10)

    #     for idx, (generated, reference) in enumerate(zip(batch_generated, batch_references)):
    #         measured = get_measure(reference, generated)
            
    #         f1_lst.append(measured['f1'])
    #         rouge_lst.append(measured['rouge'])
    #         bleu_lst.append(measured['bleu'])
    #         bleu_rt_lst.append(measured['bleu-rt'])

    # print(f'{name}\'s evaluation : ')
    # print(f'TOTAL F1 : {sum(f1_lst)/len(f1_lst)}')
    # print(f'TOTAL ROUGE : {sum(rouge_lst)/len(rouge_lst)}')
    # print(f'TOTAL BLEU : {sum(bleu_lst)/len(bleu_lst)}')
    # print(f'TOTAL BLEU-RT : {sum(bleu_rt_lst)/len(bleu_rt_lst)}')

    # if nia_path.split('/')[-1] == 'GPT_TABLE.json':
    #     print(f'TOTAL RMS Precsion : {sum(RMS_precision_lst)/len(RMS_precision_lst)}')
    #     print(f'TOTAL RMS Recall : {sum(RMS_recall_lst)/len(RMS_recall_lst)}')
    #     print(f'TOTAL RMS F1 : {sum(RMS_f1_lst)/len(RMS_f1_lst)}')

    #     print(f'TOTAL RNSS : {sum(RNSS_lst)/len(RNSS_lst)}')

