import os
import cv2

def extract_frames(video_path, output_folder):
    # 创建与视频同名的子文件夹
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存帧为图片
        frame_name = f"{frame_count:04d}.png"  # 格式化帧名，如0000.png
        frame_path = os.path.join(video_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_name}")

def process_video_folder(input_folder, output_folder):
    # 遍历输入文件夹中的所有视频文件
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        if os.path.isfile(video_path) and video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            extract_frames(video_path, output_folder)

# 输入和输出文件夹路径
input_folder = "/mnt/private/xwy/aed-mae/data/ubnormal/videos" # 替换为你的视频文件夹路径
output_folder = "/mnt/private/xwy/aed-mae/data/ubnormal/frames" # 替换为你的输出文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 处理视频文件夹
process_video_folder(input_folder, output_folder)