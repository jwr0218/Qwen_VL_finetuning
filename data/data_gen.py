import json
from typing import List, Dict, Any

class WebtoonQueryAnswerGenerator:
    """웹툰 JSON 데이터에서 VLM 학습용 Query-Answer 쌍을 생성하는 클래스"""
    
    def __init__(self, json_data: List[Dict]):
        self.data = json_data
    
    def generate_text_detection_qa(self) -> List[Dict]:
        """텍스트 정보 추출 Task용 Query-Answer 생성"""
        qa_pairs = []
        
        for page in self.data:
            if 'scene' not in page:
                continue
                
            # 모든 텍스트와 세부 정보 추출
            text_bubbles = []
            narratives = []
            effects = []
            
            for scene_key, scene_content in page['scene'].items():
                # 말풍선 텍스트
                if 'text_bubble' in scene_content:
                    for bubble in scene_content['text_bubble']:
                        text = bubble.get('text', '')
                        if text:
                            bubble_info = {
                                "text": text,
                                "location": bubble.get('location', ''),
                                "speaker": bubble.get('speaker', ''),
                                "bubble_type": bubble.get('bubble_type', ''),
                                "emotion": bubble.get('emotion', ''),
                                "scene": scene_key
                            }
                            text_bubbles.append(bubble_info)
                
                # 나레이션 텍스트
                if 'narrative' in scene_content:
                    for narrative in scene_content['narrative']:
                        text = narrative.get('text', '')
                        if text:
                            narrative_info = {
                                "text": text,
                                "location": narrative.get('location', ''),
                                "scene": scene_key
                            }
                            narratives.append(narrative_info)
                
                # 효과음
                if 'effects' in scene_content:
                    for effect in scene_content['effects']:
                        if isinstance(effect, dict) and 'text' in effect:
                            text = effect.get('text', '')
                            if text:
                                effect_info = {
                                    "text": text,
                                    "location": effect.get('location', ''),
                                    "style": effect.get('style', ''),
                                    "scene": scene_key
                                }
                                effects.append(effect_info)
            
            # 텍스트 정보 통합 응답
            answer_data = {
                "text_bubbles": text_bubbles,
                "narratives": narratives, 
                "effects": effects
            }
            
            qa_pairs.append({
                "image_path": page.get('image', ''),
                "query": "이 웹툰 이미지의 모든 텍스트(말풍선, 나레이션, 효과음)를 추출해주세요.",
                "answer": json.dumps(answer_data, ensure_ascii=False, indent=2),
                "task_type": "text_detection"
            })
        
        return qa_pairs
    
    def generate_scene_analysis_qa(self) -> List[Dict]:
        """장면 정보 분석 Task용 Query-Answer 생성"""
        qa_pairs = []
        
        for page in self.data:
            if 'overall_description' not in page:
                continue
                
            overall_desc = page['overall_description']
            previous_desc = page.get('previous_description')
            
            # 1. Glossary 추출 QA
            glossary = overall_desc.get('glossary', {})
            qa_pairs.append({
                "image_path": page.get('image'),
                "query": f"이전 상황: {previous_desc}\n\n이 웹툰 이미지의 특별한 용어나 고유명사를 추출해주세요.",
                "answer": json.dumps({"glossary": glossary}, ensure_ascii=False, indent=2),
                "task_type": "glossary_extraction"
            })
            
            # 2. Overall Description 생성 QA
            overall_text = overall_desc.get('text')
            qa_pairs.append({
                "image_path": page.get('image'),
                "query": f"이전 상황: {previous_desc}\n\n이 웹툰 페이지의 전체 상황을 설명해주세요.",
                "answer": overall_text,
                "task_type": "overall_description"
            })
            
        return qa_pairs
    
    def generate_all_in_one_qa(self) -> List[Dict]:
        """모든 정보를 한 번의 추론으로 추출하는 완전 통합 Task"""
        qa_pairs = []
        
        for page in self.data:
            # 텍스트 정보 추출
            text_bubbles = []
            narratives = []
            effects = []
            
            if 'scene' in page:
                for scene_key, scene_content in page['scene'].items():
                    # 말풍선 텍스트
                    if 'text_bubble' in scene_content:
                        for bubble in scene_content['text_bubble']:
                            text = bubble.get('text', '')
                            if text:
                                bubble_info = {
                                    "text": text,
                                    "location": bubble.get('location', ''),
                                    "speaker": bubble.get('speaker', ''),
                                    "bubble_type": bubble.get('bubble_type', ''),
                                    "emotion": bubble.get('emotion', ''),
                                    "scene": scene_key
                                }
                                text_bubbles.append(bubble_info)
                    
                    # 나레이션 텍스트
                    if 'narrative' in scene_content:
                        for narrative in scene_content['narrative']:
                            text = narrative.get('text', '')
                            if text:
                                narrative_info = {
                                    "text": text,
                                    "location": narrative.get('location', ''),
                                    "scene": scene_key
                                }
                                narratives.append(narrative_info)
                    
                    # 효과음
                    if 'effects' in scene_content:
                        for effect in scene_content['effects']:
                            if isinstance(effect, dict) and 'text' in effect:
                                text = effect.get('text', '')
                                if text:
                                    effect_info = {
                                        "text": text,
                                        "location": effect.get('location', ''),
                                        "style": effect.get('style', ''),
                                        "scene": scene_key
                                    }
                                    effects.append(effect_info)
            
            # 장면 정보 추출
            overall_desc = page.get('overall_description', {})
            previous_desc = page.get('previous_description', '')
            
            # 완전 통합 응답 데이터
            complete_analysis = {
                "text_information": {
                    "text_bubbles": text_bubbles,
                    "narratives": narratives,
                    "effects": effects
                },
                "scene_information": {
                    "overall_description": overall_desc.get('text', ''),
                    "glossary": overall_desc.get('glossary', {}),
                    "scene_analysis": overall_desc.get('scene_analysis', {})
                }
            }
            
            qa_pairs.append({
                "image_path": page.get('image', ''),
                "query": f"이전 상황: {previous_desc}\n\n이 웹툰 이미지를 완전히 분석해서 JSON 형식으로 응답해주세요.",
                "answer": json.dumps(complete_analysis, ensure_ascii=False, indent=2),
                "task_type": "complete_analysis"
            })
        
        return qa_pairs
    
    def generate_all_tasks_qa(self) -> Dict[str, List[Dict]]:
        """모든 Task의 Query-Answer 쌍을 생성"""
        return {
            "text_detection": self.generate_text_detection_qa(),
            "scene_analysis": self.generate_scene_analysis_qa(),
            "complete_analysis": self.generate_all_in_one_qa()
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
    
    combined_path = f"{output_dir}/webtoon_all_combined_qa.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_combined, f, ensure_ascii=False, indent=2)
    
    stats['all_combined'] = len(all_combined)
    print(f"전체 통합 데이터셋: {len(all_combined)}개 QA 쌍 생성 -> {combined_path}")
    
    return stats

def create_balanced_training_data(json_path: str, output_path: str = "webtoon_balanced_training.json"):
    """
    균형잡힌 학습 데이터를 생성합니다.
    
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
    
    # 모든 데이터를 하나로 합치고 셔플
    mixed_data = []
    for task_name, qa_list in all_qa_data.items():
        mixed_data.extend(qa_list)
    
    # 데이터 셔플
    import random
    random.seed(42)
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
    json_path = "/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_total.json"
    
    # 1. Task별로 분리된 QA 데이터 생성
    print("=== Task별 분리 데이터 생성 ===")
    stats = create_training_data(json_path, output_dir="/workspace/Toonspace_VLM/data/grok_json_file")
    
    # 2. 균형잡힌 혼합 데이터 생성
    print("\n=== 균형잡힌 혼합 데이터 생성 ===")
    mixed_data = create_balanced_training_data(
        json_path, 
        output_path="/workspace/Toonspace_VLM/data/grok_json_file/webtoon_balanced_training.json"
    )
    
    print("\n=== 생성 완료 ===")
    print("생성된 파일들:")
    print("- webtoon_text_detection_qa.json: 텍스트 정보 추출 (말풍선, 나레이션, 효과음)")
    print("- webtoon_scene_analysis_qa.json: 장면 정보 분석 (용어집, 전체 설명, 패널별 분석)")
    print("- webtoon_comprehensive_qa.json: 통합 분석 (텍스트 + 장면 정보)")
    print("- webtoon_all_combined_qa.json: 전체 통합")
    print("- webtoon_balanced_training.json: 균형잡힌 혼합 (추천)")
    
    print(f"\n총 {sum(stats.values())}개의 QA 쌍이 생성되었습니다!")
    
    # 생성된 데이터 샘플 확인
    print("\n=== 데이터 샘플 확인 ===")
    if mixed_data:
        print("텍스트 정보 추출 샘플:")
        text_samples = [item for item in mixed_data if item['task_type'] == 'text_detection']
        if text_samples:
            print(f"Query: {text_samples[0]['query'][:100]}...")
            print(f"Answer (일부): {text_samples[0]['answer'][:200]}...")
        
        print("\n완전 통합 분석 샘플:")
        complete_samples = [item for item in mixed_data if item['task_type'] == 'complete_analysis']
        if complete_samples:
            print(f"Query: {complete_samples[0]['query'][:150]}...")
            print(f"Answer (일부): {complete_samples[0]['answer'][:300]}...")