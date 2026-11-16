## Day 03 - Joint Angle, Normalization & Frame Feature

### 오늘 배운 내용

- COCO 17 Keypoint의 좌표, Confidence와 Left / Right 기준
- 세 Keypoint를 이용한 팔꿈치·무릎 각도 계산
- Hip Center 기준 중심 이동과 Shoulder Width 기준 크기 정규화
- Near / Far 및 Person A / Person B 비교
- Keypoint 좌우 반전과 좌우 관절 인덱스 교환
- 정규화 좌표 34개와 관절각도 4개를 합친 38-D Frame Feature 생성
- 동작에 맞는 Feature 선택과 영상에서의 시간 변화 확인

### Day 03 핵심 흐름

```text
이미지 또는 영상 Frame
        ↓
YOLO Pose
        ↓
COCO 17 Keypoint + Confidence
        ↓
좌표 정규화 + 관절각도 계산
        ↓
정규화 좌표 34개 + 관절각도 4개
        ↓
38-D Frame Feature
        ↓
영상의 각 Frame에 반복 적용
        ↓
시간에 따른 Feature 변화 확인
        ↓
Sequence 구성에 사용할 데이터
```

### 오늘 추가한 주요 코드

```text
src/
└── feature_utils.py

scripts/
├── 11_check_day03_data.py
├── 12_joint_angle.py
├── 13_normalize_pose.py
├── 14_compare_normalization.py
├── 15_flip_keypoints.py
├── 16_make_frame_feature.py
├── 17_arms_up_rule.py
├── 18_video_frame_features_forppj.py
└── 19_pose_video_quality_day03.py
```

팀 영상 처리에는 다음 두 파일을 사용했다.

- `18_video_frame_features_forppj.py`: Frame별 관절 정규화 및 38-D Feature CSV 저장
- `19_pose_video_quality_day03.py`: Skeleton 영상, Frame별 품질, 관절별 좌표와 Confidence 저장

### 오늘 이해한 핵심

관절의 Pixel 좌표는 사람이 화면 어디에 있는지, 얼마나 크게 보이는지에 따라 달라진다. 이를 줄이기 위해 모든 관절 좌표에서 골반 중심을 빼고, 양쪽 어깨 사이 거리로 나눴다.

정규화 좌표는 몸을 기준으로 관절이 어디에 있는지를 나타내고, 관절각도는 얼마나 굽혀져 있는지를 나타낸다. 두 정보를 합쳐 한 Frame을 38개의 숫자로 표현했다.

동작을 비교할 때는 각도 차이가 가장 큰 관절만 고르는 것이 아니라, 그 동작을 직접 설명할 수 있는 Feature인지도 봐야 했다. 점프 스쿼트에서는 오른쪽 팔꿈치의 각도 차이가 가장 컸지만, 하단 자세와 선 자세를 비교하는 데는 양쪽 무릎 각도가 더 직접적이었다.

또한 Feature가 계산됐다는 것과 관절이 정확하게 검출됐다는 것은 다르다. 각도 계산에 사용된 관절의 Confidence를 따로 확인해야 한다.

### Mini Challenge

팀에서 준비한 버피(A001), 점프 스쿼트(A008), 런지(A014) 영상 중 `A008_t1.mp4`를 보고서의 분석 대상으로 선택했다.

2.0초의 스쿼트 하단 자세를 Target, 2.5초의 무릎을 편 자세를 Contrast로 정했다. 두 Frame을 이미지로 저장해 관절각도를 비교하고, 같은 Target 이미지의 원본과 크롭·확대본으로 정규화 결과를 비교했다.

| 항목 | 결과 |
|---|---|
| 분석 영상 | `A008_t1.mp4` |
| 전체 Frame | 309개 |
| 정규화 계산 성공 | 309개 |
| 정규화 기준인 양쪽 어깨·골반 Confidence 통과 | 309개 |
| Frame당 Feature | 정규화 좌표 34개 + 관절각도 4개 |
| 주요 Feature | 왼쪽·오른쪽 무릎 각도 |
| 보조 Feature | 오른쪽 팔꿈치 각도 |

영상에서 왼쪽 무릎 각도는 시작 시점의 169.77°에서 Target 시점에 89.66°로 작아졌고, Contrast 시점에는 177.34°로 커졌다. 오른쪽 무릎도 164.65° → 75.10° → 176.74°로 변했다.

정규화 비교에서 Near와 Far의 어깨 폭은 각각 184.47 px, 115.50 px였다. 대응 관절의 평균 거리는 Pixel 좌표에서 90.7736 px, 정규화 좌표에서 0.1562로 계산됐다. 두 값은 단위가 달라 직접적인 감소율로 비교하지 않았다. 실제 거리를 바꿔 촬영한 실험이 아니라 동일 이미지의 크기를 조절한 비교라는 점도 보고서에 기록했다.

Confidence 0.50 미만인 관절은 오른쪽 귀 309건, 오른쪽 팔꿈치 45건, 오른쪽 눈 17건, 오른쪽 손목 11건이었다. 모든 Frame에 LOW 관절이 있었지만, 어떤 Feature를 계산하는지에 따라 확인해야 할 관절은 달랐다.

상세 결과는 `reports/day03/challenge/feature_design.md`에 정리했다.

### 다음 수업

4일차에는 Frame별 Feature를 시간 순서로 묶어 Sequence를 만든다. 한 장면의 자세를 비교하는 것에서 나아가, 동작이 어떤 순서로 변하는지 분석한다.