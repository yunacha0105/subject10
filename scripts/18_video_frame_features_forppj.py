
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device
from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)
from src.pose_utils import COCO_KEYPOINT_NAMES


# COCO 17 좌표 34개 + 관절각도 4개
ANGLE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
]

COORD_COLUMNS = [
    f"{name}_{axis}"
    for name in COCO_KEYPOINT_NAMES
    for axis in ("x", "y")
]

FEATURE_COLUMNS = COORD_COLUMNS + ANGLE_COLUMNS

META_COLUMNS = [
    "frame_id",
    "time_sec",
    "people",
    "selected_person_index",
    "box_conf",
    "valid_pose",
    "mean_keypoint_conf",
    "low_keypoint_count",
    "critical_keypoint_pass",
    "all_keypoints_pass",
    "shoulder_width_px",
]

# 정규화 기준이 되는 양쪽 어깨와 골반
CRITICAL_IDS = (5, 6, 11, 12)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MP4 Frame별 관절 정규화 및 38-D Feature 추출"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="입력 MP4 영상 경로",
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=None,
        help="Keypoint Confidence 기준",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="최대 Frame 수. 0이면 전체 영상",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/day03/tables/test",
        help="CSV 저장 폴더",
    )

    parser.add_argument(
        "--save-frames-every",
        type=int,
        default=0,
        help="검사용 JPG 저장 간격. 0이면 저장 안 함",
    )

    return parser.parse_args()


def analyze_frame(
    frame,
    model,
    config,
    device,
    threshold,
    frame_id,
    time_sec,
):
    # Frame 결과를 저장할 기본 행 생성
    row = {
        key: ""
        for key in META_COLUMNS + FEATURE_COLUMNS
    }

    row.update({
        "frame_id": frame_id,
        "time_sec": round(time_sec, 6)
        if time_sec is not None else "",
        "people": 0,
        "valid_pose": 0,
    })

    # YOLO Pose 추론
    result = model.predict(
        source=frame,
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )[0]

    # 관절이 검출되지 않았으면 기본 행 반환
    if (
        result.boxes is None
        or result.keypoints is None
        or result.keypoints.xy is None
    ):
        return row, result

    people = len(result.boxes)
    row["people"] = people

    if people == 0:
        return row, result

    # 사람별 BBox Confidence와 Keypoint 좌표
    box_conf = result.boxes.conf.cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()

    if xy.shape != (people, 17, 2):
        return row, result

    # 가장 높은 BBox Confidence를 가진 사람 선택
    main_id = int(np.argmax(box_conf))
    points = xy[main_id]

    row["selected_person_index"] = main_id
    row["box_conf"] = round(
        float(box_conf[main_id]), 5
    )

    # 관절별 Confidence 확인
    if result.keypoints.conf is not None:
        scores = result.keypoints.conf.cpu().numpy()

        if scores.shape == (people, 17):
            conf = scores[main_id]

            row["mean_keypoint_conf"] = round(
                float(np.mean(conf)), 5
            )

            row["low_keypoint_count"] = int(
                np.sum(conf < threshold)
            )

            row["critical_keypoint_pass"] = int(
                bool(
                    np.all(
                        conf[list(CRITICAL_IDS)] >= threshold
                    )
                )
            )

            row["all_keypoints_pass"] = int(
                bool(np.all(conf >= threshold))
            )

    if not np.all(np.isfinite(points)):
        return row, result

    try:
        # 골반 중심 이동 + 어깨 폭 정규화
        normalized, center, scale = (
            normalize_by_shoulder_width(points)
        )

        # 팔꿈치 2개 + 무릎 2개 각도
        angles = selected_joint_angles(points)

        # 정규화 좌표 34개
        values = (
            np.asarray(normalized, dtype=float)
            .reshape(-1)
            .tolist()
        )

        # 관절각도 4개 추가
        values += [
            float(angles[name])
            for name in ANGLE_COLUMNS
        ]

    except (
        ValueError,
        ZeroDivisionError,
        FloatingPointError,
        IndexError,
        KeyError,
    ):
        return row, result

    # 38개 Feature 정상 여부 확인
    if (
        len(values) != 38
        or not all(math.isfinite(v) for v in values)
    ):
        return row, result

    row["valid_pose"] = 1
    row["shoulder_width_px"] = round(
        float(scale), 4
    )

    for key, value in zip(FEATURE_COLUMNS, values):
        row[key] = round(float(value), 6)

    return row, result


def save_check_frames(
    frame,
    result,
    source_stem,
    frame_id,
):
    # 검사용 Frame 및 Skeleton 이미지 저장
    folder = (
        ROOT
        / "reports"
        / "day03"
        / "frames"
        / source_stem
    )

    folder.mkdir(parents=True, exist_ok=True)

    raw_path = folder / f"frame_{frame_id:06d}.jpg"
    pose_path = folder / f"pose_{frame_id:06d}.jpg"

    if not cv2.imwrite(str(raw_path), frame):
        raise RuntimeError(
            f"Frame 저장 실패: {raw_path}"
        )

    if not cv2.imwrite(str(pose_path), result.plot()):
        raise RuntimeError(
            f"Pose 이미지 저장 실패: {pose_path}"
        )


def main():
    args = parse_args()

    if (
        args.max_frames < 0
        or args.save_frames_every < 0
    ):
        raise ValueError(
            "Frame 옵션은 0 이상이어야 합니다."
        )

    source = Path(args.source)

    if not source.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {source}"
        )

    # 프로젝트 공통 설정
    config = load_config()

    kp_threshold = float(
        config.get("keypoint_confidence", 0.50)
        if args.kp_conf is None
        else args.kp_conf
    )

    if not 0.0 <= kp_threshold <= 1.0:
        raise ValueError(
            "Confidence는 0~1 범위여야 합니다."
        )

    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    # CSV 저장 위치
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = (
        output_dir
        / f"{source.stem}_frame_features.csv"
    )

    # 영상 열기
    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: {source}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    processed = 0
    numeric_valid = 0
    critical_pass = 0

    try:
        with output_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=META_COLUMNS + FEATURE_COLUMNS,
            )

            writer.writeheader()

            # Frame을 한 장씩 읽어서 정규화
            while True:
                success, frame = capture.read()

                if not success:
                    break

                processed += 1

                time_sec = (
                    (processed - 1) / fps
                    if fps > 0
                    else None
                )

                row, result = analyze_frame(
                    frame,
                    model,
                    config,
                    device,
                    kp_threshold,
                    processed,
                    time_sec,
                )

                # 현재 Frame의 Feature 저장
                writer.writerow(row)

                numeric_valid += int(
                    row["valid_pose"] == 1
                )

                critical_pass += int(
                    row["critical_keypoint_pass"] == 1
                )

                # 검사용 JPG 저장
                if (
                    args.save_frames_every > 0
                    and (
                        processed == 1
                        or processed
                        % args.save_frames_every == 0
                    )
                ):
                    save_check_frames(
                        frame,
                        result,
                        source.stem,
                        processed,
                    )

                if processed % 100 == 0:
                    print(
                        f"{source.name}: "
                        f"{processed} Frame 처리",
                        flush=True,
                    )

                # 지정 Frame 수에 도달하면 종료
                if (
                    args.max_frames > 0
                    and processed >= args.max_frames
                ):
                    break

    finally:
        capture.release()

    # 최종 결과 출력
    print(f"Video: {source.name}")
    print(f"Processed frames: {processed}")
    print(f"Numerically valid frames: {numeric_valid}")
    print(f"Critical keypoint pass: {critical_pass}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"CSV saved: {output_csv}")

    if args.save_frames_every > 0:
        print(
            "Check JPGs: "
            f"reports/day03/frames/{source.stem}/"
        )


if __name__ == "__main__":
    main()
