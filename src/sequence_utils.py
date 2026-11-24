from __future__ import annotations


# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
# NumPy:
# 관절좌표 계산, 배열 생성, 시간축 보간에 사용합니다.
#
# Pandas:
# 5일차 Keypoint CSV의 Frame 데이터를 처리합니다.

import numpy as np
import pandas as pd


# ==================================================
# [02] 이전 수업에서 만든 공통 함수 가져오기
# ==================================================
# feature_utils.py:
# 관절좌표 정규화와 관절각도를 계산합니다.
#
# pose_utils.py:
# COCO 형식의 17개 관절 이름을 가져옵니다.
#
# 기존 함수를 재사용하므로 같은 계산 기준을 유지합니다.

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# [03] 38-D Frame Feature 이름 정의
# ==================================================
# COCO 17개 관절마다 x, y 좌표를 사용합니다.
#
# 17개 관절 × 2개 좌표 = 34개
# 관절각도 4개 추가 = 총 38개
#
# 이 이름들은 나중에 Sequence CSV의
# Feature 컬럼 이름을 만드는 데 사용합니다.

FRAME_FEATURE_NAMES = []

for name in COCO_KEYPOINT_NAMES:
    FRAME_FEATURE_NAMES.extend(
        [
            f"{name}_x",
            f"{name}_y",
        ]
    )

FRAME_FEATURE_NAMES.extend(
    [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
    ]
)


# ==================================================
# [04] Keypoint CSV 한 행에서 17개 관절좌표 추출
# ==================================================
# 5일차 CSV의 한 행은 일반적으로 한 Frame입니다.
#
# 각 관절의 Pixel 좌표를 읽습니다.
#
# 예:
# nose_x_px, nose_y_px
# left_shoulder_x_px, left_shoulder_y_px
#
# 좌표가 없거나 유효하지 않으면 None을 반환합니다.
#
# 정상 결과:
# (17, 2) 형태의 NumPy 배열

def keypoints_from_row(
    row: pd.Series,
):
    keypoints = []

    for name in COCO_KEYPOINT_NAMES:
        x = row.get(
            f"{name}_x_px"
        )

        y = row.get(
            f"{name}_y_px"
        )

        if (
            pd.isna(x)
            or pd.isna(y)
        ):
            return None

        keypoints.append(
            [
                float(x),
                float(y),
            ]
        )

    values = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(values)
    ):
        return None

    return values


# ==================================================
# [05] 현재 Frame에서 Pose가 검출되었는지 확인
# ==================================================
# pose_detected:
# 5일차 Keypoint CSV에 저장된 검출 결과입니다.
#
# 1 = Pose 검출
# 0 = Pose 미검출
#
# Pose가 검출되지 않았다면 해당 Frame을 제외합니다.
#
# 검출된 Frame에서는 [04] 함수를 호출하여
# 실제 17개 관절좌표를 가져옵니다.

def frame_feature_from_row(
    row: pd.Series,
):
    pose_detected = int(
        row.get(
            "pose_detected",
            0,
        )
    )

    if pose_detected != 1:
        return None

    keypoints = keypoints_from_row(
        row
    )

    if keypoints is None:
        return None


    # ==============================================
    # [06] 관절좌표 정규화 및 관절각도 계산
    # ==============================================
    # normalize_by_shoulder_width():
    # 기존 수업에서 만든 정규화 함수를 사용합니다.
    #
    # selected_joint_angles():
    # 왼쪽·오른쪽 팔꿈치와 무릎 각도를 계산합니다.
    #
    # 정규화할 수 없는 Frame은 제외합니다.

    try:
        normalized, _, _ = (
            normalize_by_shoulder_width(
                keypoints
            )
        )
    except ValueError:
        return None

    angles = (
        selected_joint_angles(
            keypoints
        )
    )


    # ==============================================
    # [07] 한 Frame을 38개 Feature로 변환
    # ==============================================
    # 정규화 좌표:
    # 17개 관절 × (x, y) = 34개
    #
    # 관절각도:
    # 팔꿈치 2개 + 무릎 2개 = 4개
    #
    # 총 38개 Feature를 생성합니다.
    #
    # Feature 개수와 유효성을 검사한 후 반환합니다.

    values = []

    for point in normalized:
        values.extend(
            [
                float(point[0]),
                float(point[1]),
            ]
        )

    values.extend(
        [
            float(
                angles[
                    "left_elbow_angle"
                ]
            ),
            float(
                angles[
                    "right_elbow_angle"
                ]
            ),
            float(
                angles[
                    "left_knee_angle"
                ]
            ),
            float(
                angles[
                    "right_knee_angle"
                ]
            ),
        ]
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(values) != 38:
        raise RuntimeError(
            "Frame Feature가 "
            "38개가 아닙니다."
        )

    if not np.all(
        np.isfinite(values)
    ):
        return None

    return values


# ==================================================
# [08] Keypoint CSV를 Frame 번호 순서로 정렬
# ==================================================
# 5일차 Keypoint CSV 전체를 입력받습니다.
#
# frame_id 컬럼이 있는지 확인합니다.
#
# 시간 순서를 유지하기 위해
# frame_id를 기준으로 오름차순 정렬합니다.
#
# 이후 모든 Frame을 순서대로 처리합니다.

def build_frame_feature_matrix(
    df: pd.DataFrame,
):
    if "frame_id" not in df.columns:
        raise RuntimeError(
            "frame_id 열이 없습니다."
        )

    ordered = (
        df.sort_values(
            "frame_id"
        )
        .reset_index(
            drop=True
        )
    )


    # ==============================================
    # [09] 모든 Frame을 38-D Feature로 변환
    # ==============================================
    # Frame을 하나씩 읽어 [05] 함수를 호출합니다.
    #
    # Pose가 검출되지 않은 Frame은 제외합니다.
    #
    # 정상 Frame의 Feature와 frame_id만 저장합니다.
    #
    # features:
    # Frame별 38개 Feature
    #
    # frame_ids:
    # 각 Feature에 대응하는 원본 Frame 번호

    features = []
    frame_ids = []

    for _, row in (
        ordered.iterrows()
    ):
        feature = (
            frame_feature_from_row(
                row
            )
        )

        if feature is None:
            continue

        features.append(feature)

        frame_ids.append(
            int(row["frame_id"])
        )


    # ==============================================
    # [10] 유효 Frame이 없을 경우 처리
    # ==============================================
    # 유효 Frame이 하나도 없다면
    # 빈 Feature 배열과 빈 Frame ID 배열을 반환합니다.
    #
    # 정상 Frame이 있다면:
    #
    # features → (유효 Frame 수, 38)
    # frame_ids → (유효 Frame 수,)
    #
    # 두 배열을 함께 반환합니다.

    if not features:
        return (
            np.empty(
                (
                    0,
                    len(
                        FRAME_FEATURE_NAMES
                    ),
                ),
                dtype=np.float32,
            ),
            np.asarray(
                [],
                dtype=np.int32,
            ),
        )

    return (
        np.stack(features),
        np.asarray(
            frame_ids,
            dtype=np.int32,
        ),
    )


# ==================================================
# [11] Sequence 변환 전 입력 데이터 검사
# ==================================================
# 영상마다 Frame 수가 다르기 때문에
# 동일한 30 Step으로 변환해야 합니다.
#
# 입력:
# frame_features → (T, 38)
# frame_ids      → 실제 Frame 번호
#
# 다음 조건을 검사합니다.
#
# 1. Feature 배열이 2차원인가?
# 2. Feature 수와 Frame ID 수가 같은가?
# 3. 유효 Frame이 최소 2개인가?
# 4. 목표 Step 수가 2 이상인가?
# 5. Frame 시간 범위가 정상인가?

def resample_sequence(
    frame_features,
    frame_ids,
    target_steps: int = 30,
):
    frame_features = np.asarray(
        frame_features,
        dtype=np.float32,
    )

    frame_ids = np.asarray(
        frame_ids,
        dtype=np.float32,
    )

    if frame_features.ndim != 2:
        raise ValueError(
            "frame_features는 "
            "(T, F) 형태여야 합니다."
        )

    if len(frame_features) != len(
        frame_ids
    ):
        raise ValueError(
            "Feature 수와 Frame ID 수가 "
            "다릅니다."
        )

    if len(frame_features) < 2:
        raise ValueError(
            "유효한 Frame Feature가 "
            "2개 이상 필요합니다."
        )

    if target_steps < 2:
        raise ValueError(
            "target_steps는 "
            "2 이상이어야 합니다."
        )

    span = (
        frame_ids[-1]
        - frame_ids[0]
    )

    if span <= 0:
        raise ValueError(
            "Frame 시간 순서를 "
            "확인할 수 없습니다."
        )


    # ==============================================
    # [12] 원본 Frame의 시간축을 0~1로 정규화
    # ==============================================
    # 영상마다 전체 Frame 수가 다릅니다.
    #
    # 원본 Frame의 시작을 0,
    # 마지막을 1로 맞춥니다.
    #
    # target_x:
    # 시작(0)부터 끝(1)까지
    # 동일한 간격의 30개 시간 지점을 만듭니다.
    #
    # 이렇게 하면 영상 길이가 달라도
    # 동일한 시간축 기준으로 비교할 수 있습니다.

    original_x = (
        frame_ids
        - frame_ids[0]
    ) / span

    target_x = np.linspace(
        0.0,
        1.0,
        target_steps,
    )


    # ==============================================
    # [13] 보간을 이용하여 30 Step Sequence 생성
    # ==============================================
    # 각 Feature의 시간에 따른 값을 보간합니다.
    #
    # np.interp():
    # 기존 Frame 사이의 값을 계산하여
    # 새로운 시간 지점의 Feature를 만듭니다.
    #
    # 예:
    # 85 Frame  → 30 Step
    # 125 Frame → 30 Step
    # 266 Frame → 30 Step
    #
    # 최종 출력 형태:
    # (30, 38)

    output = np.empty(
        (
            target_steps,
            frame_features.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(
        frame_features.shape[1]
    ):
        output[
            :,
            feature_index
        ] = np.interp(
            target_x,
            original_x,
            frame_features[
                :,
                feature_index,
            ],
        )

    return output


# ==================================================
# [14] 30 × 38 Sequence를 1차원 Feature로 변환
# ==================================================
# Random Forest에 입력하기 위해
# 2차원 Sequence 배열을 1차원으로 펼칩니다.
#
# 변환 전:
# (30, 38)
#
# 변환 후:
# (1140,)
#
# Clip 하나가 1,140개 숫자로 표현됩니다.

def flatten_sequence(
    sequence,
):
    return np.asarray(
        sequence,
        dtype=np.float32,
    ).reshape(-1)


# ==================================================
# [15] Sequence Feature의 CSV 컬럼 이름 생성
# ==================================================
# 각 시간 Step과 Feature 이름을 결합합니다.
#
# 예:
# t00_nose_x
# t00_nose_y
# ...
# t29_right_knee_angle
#
# 30 Step × 38 Feature
# = 총 1,140개 컬럼 이름을 생성합니다.
#
# 실제 34번 코드에서 CSV를 만들 때 사용합니다.

def sequence_feature_names(
    target_steps: int,
):
    names = []

    for step in range(
        target_steps
    ):
        for feature_name in (
            FRAME_FEATURE_NAMES
        ):
            names.append(
                f"t{step:02d}_"
                f"{feature_name}"
            )

    return names


# ==================================================
# [16] Static 비교용 가운데 Frame Feature 선택
# ==================================================
# 6일차 38번 비교 실험에서 사용합니다.
#
# 전체 Sequence를 사용하는 대신
# 유효 Frame 중 가운데 Frame 하나를 선택합니다.
#
# Static:
# 가운데 Frame의 38개 Feature
#
# Sequence:
# 전체 동작의 1,140개 Feature
#
# 두 방법의 성능을 비교하기 위한 함수입니다.

def middle_frame_feature(
    frame_features,
):
    frame_features = np.asarray(
        frame_features,
        dtype=np.float32,
    )

    if len(frame_features) == 0:
        raise ValueError(
            "Frame Feature가 없습니다."
        )

    middle_index = (
        len(frame_features)
        // 2
    )

    return frame_features[
        middle_index
    ]