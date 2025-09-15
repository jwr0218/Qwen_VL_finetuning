import json

# JSON 파일 읽기
with open('/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_english.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

with open('/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_japan.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# 리스트 합치기
merged_data = data1 + data2

# 합친 결과를 새 파일로 저장
with open('/workspace/Toonspace_VLM/data/Webtoon_Narrative_Dialogue_total.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"총 {len(merged_data)}개 항목이 합쳐졌습니다.")