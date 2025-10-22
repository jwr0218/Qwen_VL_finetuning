import json

def extract_webtoon_data_separated(input_data):
    """
    웹툰 JSON 데이터에서 OCR, 위치, 설명 정보를 추출하여 분리된 형태로 변환
    - OCR&BBOX 데이터와 Description 데이터를 별도로 생성
    
    Args:
        input_data: 원본 JSON 데이터 (리스트 또는 딕셔너리)
    
    Returns:
        list: 변환된 데이터 리스트 (OCR&BBOX와 Description이 각각 별도 항목으로)
    """
    result = []
    
    # 입력 데이터가 리스트가 아닌 경우 리스트로 변환
    if not isinstance(input_data, list):
        input_data = [input_data]
    
    for item in input_data:
        # 기본 정보 추출
        image_path = item.get('image', '')
        description = item.get('overall_description', {}).get('text', '')
        try:
            previous_desc = item.get('previous_description', {}).get('text', '')
        except : 
            previous_desc = '[Empty]'
        
        # OCR 데이터를 문자열로 결합
        ocr_texts = []
        scene_data = item.get('scene', {})
        
        # 각 scene을 순회하며 OCR 정보 수집
        for scene_key, scene_value in scene_data.items():
            # effects에서 텍스트 추출
            effects = scene_value.get('effects', [])
            for effect in effects:
                if 'text' in effect and 'location' in effect:
                    ocr_texts.append(f"{effect['text']} : {effect['location']}")
            
            # text_bubble에서 텍스트 추출
            text_bubbles = scene_value.get('text_bubble', [])
            for bubble in text_bubbles:
                if 'text' in bubble and 'location' in bubble:
                    ocr_texts.append(f"{bubble['text']} : {bubble['location']}")
            
            # narrative에서 텍스트 추출 (만약 위치 정보가 있다면)
            narratives = scene_value.get('narrative', [])
            for narrative in narratives:
                if isinstance(narrative, dict) and 'text' in narrative and 'location' in narrative:
                    ocr_texts.append(f"{narrative['text']} : {narrative['location']}")
        
        # OCR 텍스트들을 하나의 문자열로 결합
        ocr_label = "\n".join(ocr_texts) if ocr_texts else ""
        
        # OCR&BBOX 데이터 생성 (OCR 데이터가 있는 경우만)
        if ocr_label.strip():
            ocr_item = {
                "image_path": image_path,
                "query": "OCR&BBOX",
                "answer": ocr_label
            }
            result.append(ocr_item)
        
        # Description 데이터 생성 (설명이 있는 경우만)
        if description.strip():
            desc_item = {
                "image_path": image_path,
                "query": f"previous : {previous_desc} \n order : Description",
                "answer": description
            }
            result.append(desc_item)
    
    return result


def process_webtoon_file_separated(input_file_path, output_file_path):
    """
    JSON 파일을 읽어서 데이터를 추출하고 분리된 형태로 새로운 파일에 저장
    
    Args:
        input_file_path: 입력 JSON 파일 경로
        output_file_path: 출력 JSON 파일 경로
    """
    try:
        # JSON 파일 읽기
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 데이터 추출 (분리된 형태)
        extracted_data = extract_webtoon_data_separated(data)
        
        # 결과를 새로운 JSON 파일로 저장
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(extracted_data, file, ensure_ascii=False, indent=2)
        
        # 통계 정보 출력
        ocr_count = sum(1 for item in extracted_data if item['query'] == 'OCR&BBOX')
        desc_count = sum(1 for item in extracted_data if item['query'] == 'Description')
        
        print(f"데이터 추출 완료!")
        print(f"총 {len(extracted_data)}개의 항목이 {output_file_path}에 저장되었습니다.")
        print(f"- OCR&BBOX: {ocr_count}개")
        print(f"- Description: {desc_count}개")
        
        return extracted_data
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file_path}")
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {input_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")


def preview_data_structure(input_file_path, num_samples=3):
    """
    변환된 데이터 구조를 미리 확인하는 함수
    
    Args:
        input_file_path: 입력 JSON 파일 경로
        num_samples: 미리 볼 샘플 개수
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        extracted_data = extract_webtoon_data_separated(data[:num_samples])
        
        print("=== 변환된 데이터 구조 미리보기 ===")
        for i, item in enumerate(extracted_data):
            print(f"\n[항목 {i+1}]")
            print(f"image: {item['image_path']}")
            print(f"query: {item['query']}")
            print(f"label: {item['answer'][:100]}..." if len(item['answer']) > 100 else f"answer: {item['answer']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"미리보기 중 오류 발생: {str(e)}")


# 예시 사용법
if __name__ == "__main__":
    input_path = "/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_total.json"
    output_path = "ocr_description/OCR_DESCRIPTION.json"
    
    # 데이터 구조 미리보기
    print("데이터 구조를 먼저 확인합니다...")
    preview_data_structure(input_path, num_samples=2)
    
    # 실제 변환 수행
    print("\n실제 변환을 수행합니다...")
    process_webtoon_file_separated(input_path, output_path)