from ultralytics import YOLO
import pickle
import cv2
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
import supervision as sv
import sys
import os

from utils import measure_distance

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
        self.team_colors = {}

    def put_chinese_text(self, image, text, position, font_path, font_size, color):
        # Convert image to RGB (OpenCV uses BGR by default)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        # Load font
        font = ImageFont.truetype(font_path, font_size)
        # Draw text
        draw.text(position, text, font=font, fill=color)
        # Convert image back to BGR
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

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
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # {0: ball, 1: goalkeeper, 2: player, 3: referee}
            cls_names_inv = {cls_name: i for i, cls_name in cls_names.items()}
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
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            # ball_cnt = 0
            for (
                frame_detection
            ) in detection_supervision:  # 为什么这里不用detection_with_tracks
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                # track_id = frame_detection[4]
                if cls_id == cls_names_inv["ball"]:
                    # ball_cnt += 1
                    # if ball_cnt > 1:
                    #     print(f"Frame {frame_num} has more than one ball!")
                    #     print(f"Ball {ball_cnt} bbox: {bbox}")
                    tracks["ball"][frame_num][1] = {"bbox": bbox}  # 只有一个球
                    # tracks["ball"][frame_num][track_id] = {"bbox": bbox}  # 所有球？
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
        y1 = int(bbox[1])
        x_center, _ = get_center(bbox)

        triangle_width = 10  # 0.5 * width
        triangle_height = 20
        triangle_points = np.array(
            [
                [x_center, y1],
                [x_center - 10, y1 - triangle_height],
                [x_center + 10, y1 - triangle_height],
            ]
        )

        cv2.drawContours(
            frame, [triangle_points], contourIdx=0, color=color, thickness=cv2.FILLED
        )
        cv2.drawContours(
            frame, [triangle_points], contourIdx=0, color=(0, 0, 0), thickness=2
        )

        return frame

    def draw_annotation(self, frames, tracks, team_control):
        output_frames = []
        # for frame_num, frame in enumerate(tqdm(frames, desc="Drawing Annotation")):
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            # 画 player
            for track_id, player in player_dict.items():
                color = player.get("team_color", (10, 10, 245))
                frame = self.draw_ellipse(
                    frame, player["bbox"], color=color, track_id=track_id
                )
                # 如果控球，画红色三角
                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame, player["bbox"], color=(245, 10, 10)
                    )
            # 画 referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], color=(10, 245, 245))
            # 画 ball，红色三角
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], color=(10, 10, 245))

            frame = self.draw_team_ball_control(frame, frame_num, team_control)
            output_frames.append(frame)

        return output_frames

    def interpolate_ball(self, ball_positions):
        ball_pos = [
            x.get(1, {}).get("bbox", [np.nan, np.nan, np.nan, np.nan])
            for x in ball_positions
        ]

        # 创建DataFrame时处理空值
        df_ball_pos = pd.DataFrame(ball_pos, columns=["x1", "y1", "x2", "y2"])

        # 插值并向后填充
        df_ball_pos = df_ball_pos.interpolate().bfill()
        ball_positions = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]

        # 平滑
        pos_list = []
        for frame in ball_positions:
            bbox = frame[1]["bbox"]
            pos = get_center(bbox)
            pos_list.append((pos,bbox))
        dist_list = []
        for i in range(1, len(pos_list)):
            dist = measure_distance(pos_list[i][0], pos_list[i - 1][0])
            # print(f'{dist:.2f}')
            dist_list.append(dist)
        while max(dist_list) > 73:
            for i in range(len(dist_list) - 1):
                if dist_list[i] > 73:
                    pos_list[i + 1] = pos_list[i]
                    # 重新计算当前和下一距离
                    if i + 1 < len(dist_list):
                        dist_list[i] = measure_distance(pos_list[i][0], pos_list[i + 1][0])
                    if i + 2 < len(dist_list):
                        dist_list[i + 1] = measure_distance(pos_list[i + 1][0], pos_list[i + 2][0])
        # 将pos_list付给ball_positions
        for i in range(len(ball_positions)):
            if  measure_distance(pos_list[i][0], get_center(ball_positions[i][1]["bbox"])) > 50:
                ball_positions[i][1]["bbox"] = pos_list[i][1]
        return ball_positions

    def interpolate_ball_plus(self, ball_positions):
        # 如果每一帧中只有一个球，使用原来的逻辑
        if all(len(frame_balls) <= 1 for frame_balls in ball_positions):
            ball_pos = [x.get(1, {}).get("bbox", []) for x in ball_positions]
            df_ball_pos = pd.DataFrame(ball_pos, columns=["x1", "y1", "x2", "y2"])
            # 插值
            df_ball_pos = df_ball_pos.interpolate().bfill()
            ball_positions = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]
        else:
            # 处理每一帧中可能有多个球的情况
            all_balls_pos = []
            for frame_balls in ball_positions:
                frame_ball_positions = []
                for track_id, ball in frame_balls.items():
                    frame_ball_positions.append(ball.get("bbox", []))
                all_balls_pos.append(frame_ball_positions)

            # 确保所有帧都有相同数量的球（填充空值）
            max_balls = max(len(balls) for balls in all_balls_pos)
            for balls in all_balls_pos:
                while len(balls) < max_balls:
                    balls.append([np.nan, np.nan, np.nan, np.nan])

            # 将球的位置转换为DataFrame并插值
            ball_dfs = []
            for i in range(max_balls):
                single_ball_pos = [balls[i] for balls in all_balls_pos]
                df_single_ball_pos = pd.DataFrame(
                    single_ball_pos, columns=["x1", "y1", "x2", "y2"]
                )
                df_single_ball_pos = df_single_ball_pos.interpolate().bfill()
                ball_dfs.append(df_single_ball_pos)

            # 合并所有球的位置
            interpolated_positions = []
            for row in zip(*[df.to_numpy().tolist() for df in ball_dfs]):
                frame_balls = {i + 1: {"bbox": ball} for i, ball in enumerate(row)}
                interpolated_positions.append(frame_balls)

            ball_positions = interpolated_positions

        return ball_positions

    def draw_team_ball_control(self, frame, frame_num, team_control):
        team_control = np.array(team_control)
        # 绘画透明矩形
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 965), (255, 255, 255), -1)
        alpha = 0.37
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_control_till_frame = team_control[: frame_num + 1]
        team_1 = team_control_till_frame[team_control_till_frame == 1].shape[0]
        team_2 = team_control_till_frame[team_control_till_frame == 2].shape[0]
        total = team_1 + team_2
        team_1_percent = team_1 / total if total != 0 else 0
        team_2_percent = team_2 / total if total != 0 else 0

        team_1_color = (
            int(self.team_colors[1][0]),
            int(self.team_colors[1][1]),
            int(self.team_colors[1][2]),
        )
        cv2.rectangle(frame, (1370, 880), (1390, 900), team_1_color, -1)
        cv2.putText(
            frame,
            f"Team 1 Control: {team_1_percent:.2%}",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        # frame = self.put_chinese_text(frame, f"队 1 控球率: {team_1_percent:.2%}", (1400, 900), "font/simsunb.ttf", 30, (0, 0, 0))

        team_2_color = (
            int(self.team_colors[2][0]),
            int(self.team_colors[2][1]),
            int(self.team_colors[2][2]),
        )
        cv2.rectangle(frame, (1370, 930), (1390, 950), team_2_color, -1)
        cv2.putText(
            frame,
            f"Team 2 Control: {team_2_percent:.2%}",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        # frame = self.put_chinese_text(frame, f"队 2 控球率: {team_2_percent:.2%}", (1400, 950), "font/simsunb.ttf", 30, (0, 0, 0))
        return frame
