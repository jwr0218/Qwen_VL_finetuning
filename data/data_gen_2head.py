import json

def extract_webtoon_data(input_data):
    """
    웹툰 JSON 데이터에서 OCR, 위치, 설명 정보를 추출하여 새로운 형태로 변환
    
    Args:
        input_data: 원본 JSON 데이터 (리스트 또는 딕셔너리)
    
    Returns:
        list: 변환된 데이터 리스트
    """
    result = []
    
    # 입력 데이터가 리스트가 아닌 경우 리스트로 변환
    if not isinstance(input_data, list):
        input_data = [input_data]
    
    for item in input_data:
        # 기본 정보 추출
        image_path = item.get('image', '')
        description = item.get('overall_description', {}).get('text', '')
        
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
        answer1 = "\n".join(ocr_texts) if ocr_texts else ""
        
        # 결과 딕셔너리 생성
        extracted_item = {
            "image": image_path,
            "OCR": answer1,
            "DESCRIPTION": description
        }
        
        result.append(extracted_item)
    
    return result


def process_webtoon_file(input_file_path, output_file_path):
    """
    JSON 파일을 읽어서 데이터를 추출하고 새로운 파일로 저장
    
    Args:
        input_file_path: 입력 JSON 파일 경로
        output_file_path: 출력 JSON 파일 경로
    """
    try:
        # JSON 파일 읽기
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 데이터 추출
        extracted_data = extract_webtoon_data(data)
        
        # 결과를 새로운 JSON 파일로 저장
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(extracted_data, file, ensure_ascii=False, indent=2)
        
        print(f"데이터 추출 완료! {len(extracted_data)}개의 항목이 {output_file_path}에 저장되었습니다.")
        
        return extracted_data
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file_path}")
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {input_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")


# 예시 사용법
if __name__ == "__main__":
    json_path = "/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_total.json"
    process_webtoon_file("/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_total.json", 'ocr_description/output.json')