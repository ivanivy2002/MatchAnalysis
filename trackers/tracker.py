import pickle
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import sys

sys.path.append("../")
from utils import get_center, get_width, avg_digit_fontsize


class Tracker:
    def __init__(
        self, model_pth, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2
    ):
        self.model = YOLO(model_pth)
        self.tracker = sv.ByteTrack()
        self.font = font
        self.fontScale = fontScale
        self.thickness = thickness
        self.avg_digit_width, self.avg_digit_height = avg_digit_fontsize(
            font, fontScale, thickness
        )

    # 检测帧
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            detections_batch = self.model.predict(batch, conf=0.1)
            detections += detections_batch
            # break
        return detections

    # 获取目标轨迹
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        detections = self.detect_frames(frames)

        tracks = {
            "ball": [],
            "player": [],
            # "goalkeeper": [],
            "referee": [],
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # {0:ball, 2:player, 1:goalkeeper, 3:referee}
            cls_names_inv = {cls_name: i for i, cls_name in (cls_names.items())}
            # 转为supervision Detection Format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # 将goalkeeper转为player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            # print(f'***************Detect_SV Frame={frame_num}***************')
            # print(detection_supervision)
            # 追踪
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )
            # print(f'***************Detect+Tracks Frame={frame_num}***************')
            # print(detection_with_tracks)
            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}  # 只有一个球
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):  # 画椭圆
        y_low = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_width(bbox)
        height_ratio = 0.35
        cv2.ellipse(
            frame,
            center=(x_center, y_low),
            axes=(int(width), int(width * height_ratio)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y_low - rectangle_height // 2 + 15
        y2_rect = y_low + rectangle_height // 2 + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )
            # 计算track_id有几位
            track_id_digits = len(str(track_id))
            x1_text = x_center - track_id_digits * self.avg_digit_width // 2
            y1_text = y1_rect + 15
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_text)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.fontScale,
                color=(0, 0, 0),
                thickness=self.thickness,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y1 = bbox[1]
        x_center, _ = get_center(bbox)
        width = get_width(bbox)
        triangle_width = 20  # 0.5 * width
        triangle_height = 2 * triangle_width
        triangle_points = np.array(
            [
                [x_center, y1],
                [x_center - triangle_width / 2, y1 - triangle_height],
                [x_center + triangle_width / 2, y1 - triangle_height],
            ],
            dtype=int,
        )

        cv2.drawContours(
            frame, [triangle_points], contourIdx=0, color=color, thickness=cv2.FILLED
        )
        cv2.drawContours(
            frame, [triangle_points], contourIdx=0, color=(0, 0, 0), thickness=2
        )

        return frame

    def draw_annotation(self, frames, tracks):
        output_frames = []
        # for frame_num, frame in enumerate(tqdm(frames, desc="Drawing Annotation")):
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            # 画 player
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(
                    frame, player["bbox"], color=(10, 10, 245), track_id=track_id
                )
            # 画 referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], color=(10, 245, 245))
            # 画 ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], color=(10, 245, 10))

            output_frames.append(frame)
        return output_frames
