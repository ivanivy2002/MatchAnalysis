from utils import read_video, save_video
from trackers import Tracker



def main():
    # Read Video
    video_path = 'input_video/08fd33_4.mp4'
    video_frames = read_video(video_path)
    # print(f'Number of frames: {len(video_frames)}')
    # 初始化Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='stub/track_stub.pkl')

    # 画出输出
    ## 画出对象轨迹
    output_video_frames = tracker.draw_annotation(video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()