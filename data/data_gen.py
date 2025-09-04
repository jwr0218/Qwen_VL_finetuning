import json
from typing import List, Dict, Any

class WebtoonQueryAnswerGenerator:
    """웹툰 JSON 데이터에서 VLM 학습용 Query-Answer 쌍을 생성하는 클래스"""
    
    def __init__(self, json_data: List[Dict]):
        self.data = json_data
    
    def generate_scene_description_qa(self) -> List[Dict]:
        """장면 묘사 Task용 Query-Answer 생성"""
        qa_pairs = []
        
        for page in self.data:
            if 'overall_description' not in page:
                continue
                
            description = page['overall_description']
            
            # 기본 장면 묘사 QA
            qa_pairs.append({
                "image_path": page.get('image', ''),
                "query": "이 웹툰 페이지에서 일어나고 있는 상황을 자세히 설명해주세요.",
                "answer": description.get('text', '')
            })
            
            # scene_analysis가 있으면 추가 QA 생성
            if 'scene_analysis' in description:
                qa_pairs.append({
                    "image_path": page.get('image', ''),
                    "query": "이 장면이 전체 스토리에서 어떤 역할을 하는지 분석해주세요.",
                    "answer": json.dumps(description['scene_analysis'], ensure_ascii=False, indent=2)
                })
            
            # 서사적 맥락 질문
            qa_pairs.append({
                "image_path": page.get('image', ''),
                "query": "이 웹툰 장면의 서사적 맥락과 스토리 전개를 설명해주세요.",
                "answer": f"전체 상황: {description.get('text', '')}\n\n장면 분석: {json.dumps(description.get('scene_analysis', {}), ensure_ascii=False, indent=2)}"
            })
        
        return qa_pairs
    
    def generate_text_detection_qa(self) -> List[Dict]:
        """텍스트 인식 Task용 Query-Answer 생성"""
        qa_pairs = []
        
        for page in self.data:
            if 'scene' not in page:
                continue
                
            # 모든 텍스트 추출
            all_texts = []
            detailed_info = []
            
            for scene_key, scene_content in page['scene'].items():
                # 말풍선 텍스트
                if 'text_bubble' in scene_content:
                    for bubble in scene_content['text_bubble']:
                        text = bubble.get('text', '')
                        if text:
                            all_texts.append(text)
                            detailed_info.append({
                                "type": "말풍선",
                                "text": text,
                                "speaker": bubble.get('speaker', ''),
                                "location": bubble.get('위치', ''),
                                "bubble_type": bubble.get('bubble_type', ''),
                                "emotion": bubble.get('emotion', ''),
                                "scene": scene_key
                            })
                
                # 나레이션 텍스트
                if 'narrative' in scene_content:
                    for narrative in scene_content['narrative']:
                        text = narrative.get('text', '')
                        if text:
                            all_texts.append(text)
                            detailed_info.append({
                                "type": "나레이션",
                                "text": text,
                                "location": narrative.get('위치', ''),
                                "scene": scene_key
                            })
                
                # 효과음
                if 'effects' in scene_content:
                    for effect in scene_content['effects']:
                        if isinstance(effect, dict) and 'text' in effect:
                            text = effect.get('text', '')
                            if text:
                                all_texts.append(text)
                                detailed_info.append({
                                    "type": "효과음",
                                    "text": text,
                                    "location": effect.get('위치', ''),
                                    "scene": scene_key
                                })
            
            if all_texts:
                # 전체 텍스트 인식 QA
                qa_pairs.append({
                    "image_path": page.get('image', ''),
                    "query": "이 웹툰 이미지에서 모든 텍스트를 인식하고 추출해주세요.",
                    "answer": json.dumps({"모든_텍스트": all_texts}, ensure_ascii=False, indent=2)
                })
                
                # 상세 텍스트 정보 QA
                qa_pairs.append({
                    "image_path": page.get('image', ''),
                    "query": "이 웹툰 이미지의 텍스트들을 위치 정보와 함께 상세히 분석해주세요.",
                    "answer": json.dumps({"상세_텍스트_정보": detailed_info}, ensure_ascii=False, indent=2)
                })
                
                # 말풍선만 따로 추출
                bubble_texts = [item for item in detailed_info if item["type"] == "말풍선"]
                if bubble_texts:
                    qa_pairs.append({
                        "image_path": page.get('image', ''),
                        "query": "이 웹툰 이미지의 말풍선 내용을 모두 추출해주세요.",
                        "answer": json.dumps({"말풍선_내용": bubble_texts}, ensure_ascii=False, indent=2)
                    })
        
        return qa_pairs
    
    def generate_terminology_qa(self) -> List[Dict]:
        """용어 이해 Task용 Query-Answer 생성"""
        qa_pairs = []
        
        for page in self.data:
            if 'overall_description' not in page or 'glossary' not in page['overall_description']:
                continue
                
            glossary = page['overall_description']['glossary']
            
            if glossary:
                # 전체 용어집 QA
                qa_pairs.append({
                    "image_path": page.get('image', ''),
                    "query": "이 웹툰에서 사용된 특별한 용어들과 그 의미를 설명해주세요.",
                    "answer": json.dumps(glossary, ensure_ascii=False, indent=2)
                })
                
                # 개별 용어 QA 생성
                for term, definition in glossary.items():
                    qa_pairs.append({
                        "image_path": page.get('image', ''),
                        "query": f"이 웹툰에서 '{term}'의 의미는 무엇인가요?",
                        "answer": definition
                    })
                
                # 캐릭터 용어만 따로 (인물 관련 용어가 있는 경우)
                character_terms = {k: v for k, v in glossary.items() 
                                 if any(keyword in v for keyword in ['캐릭터', '인물', '남성', '여성', '동창', '친구'])}
                if character_terms:
                    qa_pairs.append({
                        "image_path": page.get('image', ''),
                        "query": "이 웹툰에 등장하는 인물들과 관련된 용어를 설명해주세요.",
                        "answer": json.dumps(character_terms, ensure_ascii=False, indent=2)
                    })
        
        return qa_pairs
    
    def generate_character_analysis_qa(self) -> List[Dict]:
        """캐릭터 분석 Task용 Query-Answer 생성"""
        qa_pairs = []
        
        for page in self.data:
            characters_info = {}
            
            # scene에서 캐릭터 정보 추출
            if 'scene' in page:
                for scene_key, scene_content in page['scene'].items():
                    # 말풍선에서 speaker 정보 추출
                    if 'text_bubble' in scene_content:
                        for bubble in scene_content['text_bubble']:
                            speaker = bubble.get('speaker', '')
                            if speaker:
                                if speaker not in characters_info:
                                    characters_info[speaker] = {
                                        "대사": [],
                                        "감정": [],
                                        "장면": []
                                    }
                                characters_info[speaker]["대사"].append(bubble.get('text', ''))
                                characters_info[speaker]["감정"].append(bubble.get('emotion', ''))
                                characters_info[speaker]["장면"].append(scene_key)
                    
                    # 나레이션에서 캐릭터 소개 추출
                    if 'narrative' in scene_content:
                        for narrative in scene_content['narrative']:
                            text = narrative.get('text', '')
                            # 캐릭터 소개로 보이는 나레이션
                            if any(keyword in text for keyword in ['동창', '캐릭터', '인물', '본명']):
                                characters_info[f"narrative_{scene_key}"] = {
                                    "설명": text,
                                    "위치": narrative.get('위치', '')
                                }
            
            # 용어집에서 캐릭터 정보 추가
            if 'overall_description' in page and 'glossary' in page['overall_description']:
                glossary = page['overall_description']['glossary']
                for term, definition in glossary.items():
                    if any(keyword in definition for keyword in ['캐릭터', '인물', '남성', '여성']):
                        if term not in characters_info:
                            characters_info[term] = {}
                        characters_info[term]["정의"] = definition
            
            if characters_info:
                # 전체 캐릭터 분석 QA
                qa_pairs.append({
                    "image_path": page.get('image', ''),
                    "query": "이 웹툰 장면에 등장하는 캐릭터들을 분석해주세요.",
                    "answer": json.dumps(characters_info, ensure_ascii=False, indent=2)
                })
                
                # 캐릭터 감정 상태 QA
                emotion_analysis = {}
                for char, info in characters_info.items():
                    if "감정" in info and info["감정"]:
                        emotion_analysis[char] = list(set(info["감정"]))  # 중복 제거
                
                if emotion_analysis:
                    qa_pairs.append({
                        "image_path": page.get('image', ''),
                        "query": "이 장면에서 캐릭터들의 감정 상태를 분석해주세요.",
                        "answer": json.dumps(emotion_analysis, ensure_ascii=False, indent=2)
                    })
                
                # 캐릭터 대사 QA
                dialogue_analysis = {}
                for char, info in characters_info.items():
                    if "대사" in info and info["대사"]:
                        dialogue_analysis[char] = info["대사"]
                
                if dialogue_analysis:
                    qa_pairs.append({
                        "image_path": page.get('image', ''),
                        "query": "이 장면에서 각 캐릭터가 한 대사를 정리해주세요.",
                        "answer": json.dumps(dialogue_analysis, ensure_ascii=False, indent=2)
                    })
        
        return qa_pairs
    
    def generate_all_tasks_qa(self) -> Dict[str, List[Dict]]:
        """모든 Task의 Query-Answer 쌍을 생성"""
        return {
            "scene_description": self.generate_scene_description_qa(),
            "text_detection": self.generate_text_detection_qa(),
            "terminology": self.generate_terminology_qa(),
            "character_analysis": self.generate_character_analysis_qa()
        }

def create_training_data(json_path: str, output_dir: str = "."):
    """
    웹툰 JSON 파일에서 VLM 학습용 데이터를 생성합니다.
    
    Args:
        json_path (str): 웹툰 분석 결과 JSON 파일 경로
        output_dir (str): 출력 디렉토리
    
    Returns:
        Dict: 생성된 데이터 통계
    """
    
    print(f"JSON 파일 로딩 중: {json_path}")
    
    # JSON 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"로드된 페이지 수: {len(json_data)}")
    
    # Query-Answer 생성기 초기화
    generator = WebtoonQueryAnswerGenerator(json_data)
    
    # 모든 Task의 QA 생성
    all_qa_data = generator.generate_all_tasks_qa()
    
    # 각 Task별로 파일 저장
    stats = {}
    for task_name, qa_list in all_qa_data.items():
        output_path = f"{output_dir}/webtoon_{task_name}_qa.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)
        
        stats[task_name] = len(qa_list)
        print(f"{task_name}: {len(qa_list)}개 QA 쌍 생성 -> {output_path}")
    
    # 통합 데이터셋도 생성 (모든 Task 합쳐서)
    all_combined = []
    for qa_list in all_qa_data.values():
        all_combined.extend(qa_list)
    
    combined_path = f"{output_dir}/webtoon_combined_qa.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_combined, f, ensure_ascii=False, indent=2)
    
    stats['combined'] = len(all_combined)
    print(f"통합 데이터셋: {len(all_combined)}개 QA 쌍 생성 -> {combined_path}")
    
    return stats

def create_mixed_training_data(json_path: str, output_path: str = "webtoon_mixed_training.json"):
    """
    다양한 Task를 섞어서 균형잡힌 학습 데이터를 생성합니다.
    
    Args:
        json_path (str): 웹툰 분석 결과 JSON 파일 경로
        output_path (str): 출력 파일 경로
    """
    
    print(f"균형잡힌 학습 데이터 생성 중...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    generator = WebtoonQueryAnswerGenerator(json_data)
    all_qa_data = generator.generate_all_tasks_qa()
    
    # Task별 데이터 개수 확인
    print("Task별 데이터 개수:")
    for task, data in all_qa_data.items():
        print(f"  {task}: {len(data)}개")
    
    # 각 Task에서 동일한 비율로 샘플링하여 섞기
    mixed_data = []
    min_samples = min(len(data) for data in all_qa_data.values() if len(data) > 0)
    
    print(f"각 Task에서 최대 {min_samples}개씩 선택하여 균형잡힌 데이터셋 생성...")
    
    import random
    random.seed(42)  # 재현가능한 결과를 위해
    
    for task_name, qa_list in all_qa_data.items():
        if len(qa_list) > min_samples:
            # 샘플링
            sampled = random.sample(qa_list, min_samples)
        else:
            sampled = qa_list
        
        # Task 타입 정보 추가
        for item in sampled:
            item['task_type'] = task_name
        
        mixed_data.extend(sampled)
    
    # 전체 데이터 셔플
    random.shuffle(mixed_data)
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mixed_data, f, ensure_ascii=False, indent=2)
    
    print(f"균형잡힌 학습 데이터 생성 완료: {len(mixed_data)}개 -> {output_path}")
    
    # Task별 최종 분포 출력
    from collections import Counter
    task_distribution = Counter(item['task_type'] for item in mixed_data)
    print("최종 Task 분포:")
    for task, count in task_distribution.items():
        print(f"  {task}: {count}개")
    
    return mixed_data

# 사용 예시
if __name__ == "__main__":
    # 기본 사용법
    json_path = "/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_building_owner_2_tmp.json"
    
    # 1. Task별로 분리된 QA 데이터 생성
    print("=== Task별 분리 데이터 생성 ===")
    stats = create_training_data(json_path, output_dir="/workspace/Toonspace_VLM/data/json_file")
    
    # 2. 균형잡힌 혼합 데이터 생성
    print("\n=== 균형잡힌 혼합 데이터 생성 ===")
    mixed_data = create_mixed_training_data(
        json_path, 
        output_path="/workspace/Toonspace_VLM/data/json_file/webtoon_balanced_training.json"
    )
    
    print("\n=== 생성 완료 ===")
    print("생성된 파일들:")
    print("- webtoon_scene_description_qa.json: 장면 묘사")
    print("- webtoon_text_detection_qa.json: 텍스트 인식")  
    print("- webtoon_terminology_qa.json: 용어 이해")
    print("- webtoon_character_analysis_qa.json: 캐릭터 분석")
    print("- webtoon_combined_qa.json: 전체 통합")
    print("- webtoon_balanced_training.json: 균형잡힌 혼합 (추천)")
    
    print(f"\n총 {sum(stats.values())}개의 QA 쌍이 생성되었습니다!")