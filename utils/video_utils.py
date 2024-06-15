import cv2
from tqdm import tqdm

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, save_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(frames, desc='Saving Video'):
        out.write(frame)
    out.release()
    print(f'Video saved at {save_path}')

