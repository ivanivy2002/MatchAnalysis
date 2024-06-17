import cv2
import numpy as np
import time
from PIL import ImageGrab

# 截屏
def capture_screen(bbox=None):
    screen = ImageGrab.grab(bbox=bbox)
    screen_np = np.array(screen)
    frame = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    return frame

# 保存视频片段
def save_video_segment(frames, filename, fps=24):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# 分析视频片段
def analyze_video_segment(filename):
    # 调用本地的检测函数，这里是process_video
    results = process_video(filename)
    return results

# 更新显示
def update_display(frame, results):
    for player in results['players']:
        x, y, w, h = player['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, player['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    score = results['score']
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

def main():
    video_segment = []
    segment_length = 5  # seconds
    fps = 24
    frame_count = 0
    
    while True:
        frame = capture_screen(bbox=(0, 0, 1920, 1080))
        video_segment.append(frame)
        frame_count += 1
        
        if frame_count >= segment_length * fps:
            # 保存视频片段
            segment_filename = 'segment.avi'
            save_video_segment(video_segment, segment_filename, fps=fps)
            
            # 分析视频片段
            results = analyze_video_segment(segment_filename)
            
            # 更新显示
            for seg_frame in video_segment:
                seg_frame = update_display(seg_frame, results)
                cv2.imshow('Live Analysis', seg_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 清空视频片段缓存
            video_segment = []
            frame_count = 0
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1/fps)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
