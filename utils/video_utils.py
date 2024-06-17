import cv2
from tqdm import tqdm

def read_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    return frames


def save_video(frames, save_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(frames, desc='Saving Video'):
        out.write(frame)
    out.release()
    print(f'Video saved at {save_path}')

