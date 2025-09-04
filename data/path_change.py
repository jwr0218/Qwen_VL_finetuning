import json

def convert_image_paths(json_file_path, output_file_path=None):
    """
    JSON 파일에서 'image' 필드의 경로를 한국어에서 영어로 변환합니다.
    
    Args:
        json_file_path (str): 입력 JSON 파일 경로
        output_file_path (str, optional): 출력 파일 경로. None이면 원본 파일 덮어쓰기
    
    Returns:
        int: 변경된 항목 수
    """
    
    print(f"JSON 파일 로딩 중: {json_file_path}")
    
    # JSON 파일 로드
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_file_path}")
        return 0
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return 0
    
    print(f"로드된 데이터 수: {len(data)}")
    
    # 변경 카운터
    changed_count = 0
    
    # 각 항목의 image 필드 변경
    for i, item in enumerate(data):
        if 'image' in item and item['image']:
            original_path = item['image']
            # 건물주누나_2화 -> building_owner_2 변경
            if '건물주누나_2화' in original_path:
                new_path = original_path.replace('건물주누나_2화', 'building_owner_2')
                item['image'] = new_path
                changed_count += 1
                print(f"[{i+1}] 변경: {original_path} -> {new_path}")
    
    # 결과 저장
    output_path = output_file_path if output_file_path else json_file_path
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n파일 저장 완료: {output_path}")
        print(f"총 {changed_count}개 항목이 변경되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
        return 0
    
    return changed_count

def batch_convert_image_paths(file_list):
    """
    여러 JSON 파일의 이미지 경로를 일괄 변환합니다.
    
    Args:
        file_list (list): JSON 파일 경로 리스트
    
    Returns:
        dict: 파일별 변경 결과
    """
    results = {}
    
    for file_path in file_list:
        print(f"\n{'='*50}")
        print(f"처리 중: {file_path}")
        print(f"{'='*50}")
        
        changed_count = convert_image_paths(file_path)
        results[file_path] = changed_count
    
    print(f"\n{'='*50}")
    print("일괄 변환 결과:")
    print(f"{'='*50}")
    
    total_changes = 0
    for file_path, count in results.items():
        print(f"{file_path}: {count}개 변경")
        total_changes += count
    
    print(f"\n전체 {total_changes}개 항목이 변경되었습니다.")
    return results

def create_backup_and_convert(json_file_path):
    """
    백업을 생성하고 이미지 경로를 변환합니다.
    
    Args:
        json_file_path (str): JSON 파일 경로
    
    Returns:
        tuple: (백업 파일 경로, 변경된 항목 수)
    """
    import shutil
    from datetime import datetime
    
    # 백업 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{json_file_path}.backup_{timestamp}"
    
    try:
        # 백업 생성
        shutil.copy2(json_file_path, backup_path)
        print(f"백업 파일 생성: {backup_path}")
        
        # 변환 실행
        changed_count = convert_image_paths(json_file_path)
        
        return backup_path, changed_count
        
    except Exception as e:
        print(f"백업 또는 변환 중 오류 발생: {e}")
        return None, 0

def verify_changes(json_file_path):
    """
    변경 결과를 검증합니다.
    
    Args:
        json_file_path (str): 검증할 JSON 파일 경로
    
    Returns:
        dict: 검증 결과
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return {"error": str(e)}
    
    korean_count = 0
    english_count = 0
    total_images = 0
    
    for item in data:
        if 'image' in item and item['image']:
            total_images += 1
            if '건물주누나_2화' in item['image']:
                korean_count += 1
            elif 'building_owner_2' in item['image']:
                english_count += 1
    
    result = {
        "total_images": total_images,
        "korean_paths": korean_count,
        "english_paths": english_count,
        "conversion_complete": korean_count == 0
    }
    
    print(f"검증 결과:")
    print(f"- 전체 이미지: {total_images}개")
    print(f"- 한국어 경로: {korean_count}개")
    print(f"- 영어 경로: {english_count}개")
    print(f"- 변환 완료: {'예' if result['conversion_complete'] else '아니오'}")
    
    return result

# 사용 예시
if __name__ == "__main__":
    # 단일 파일 변환
    json_path = "/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_building_owner_2_tmp.json"
    
    print("=== 단일 파일 변환 ===")
    # 백업 생성 후 변환
    backup_path, changed = create_backup_and_convert(json_path)
    if backup_path:
        print(f"백업: {backup_path}")
        print(f"변경: {changed}개 항목")
    
    # 변환 결과 검증
    print("\n=== 변환 결과 검증 ===")
    verify_changes(json_path)
    
    # 여러 파일 일괄 변환 예시
    """
    print("\n=== 여러 파일 일괄 변환 ===")
    file_list = [
        "/path/to/file1.json",
        "/path/to/file2.json",
        "/path/to/file3.json"
    ]
    batch_convert_image_paths(file_list)
    """
    
    print("\n=== 변환 완료 ===")