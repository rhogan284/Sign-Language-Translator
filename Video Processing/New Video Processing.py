import cv2
import numpy as np
import json
import os


def process_video(video_info, video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return

    fps = video_info['fps']

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_path = os.path.join(output_dir, f"{video_info['video_id']}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (256, 256))

    xmin, ymin, xmax, ymax = video_info['bbox']
    diagonal = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    scale_factor = 256 / diagonal

    new_width = int((xmax - xmin) * scale_factor)
    new_height = int((ymax - ymin) * scale_factor)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        if frame_number >= video_info['frame_start'] and (
                frame_number <= video_info['frame_end'] or video_info['frame_end'] == -1):
            cropped_frame = frame[ymin:ymax, xmin:xmax]
            resized_frame = cv2.resize(cropped_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            final_frame = cv2.resize(resized_frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            out.write(final_frame)

    # Release everything if job is finished
    cap.release()
    out.release()


def find_video_info(video_id, video_details):
    for entry in video_details:
        for video in entry['instances']:
            if video['video_id'] == video_id:
                return video
    return None


def main(json_file, video_folder, output_dir):
    with open(json_file, 'r') as file:
        video_details = json.load(file)

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    for video_file in video_files:
        video_id = video_file.split('.')[0]
        video_info = find_video_info(video_id, video_details)

        if video_info:
            video_path = os.path.join(video_folder, video_file)
            process_video(video_info, video_path, output_dir)
            print(f"Processed video {video_file}.")
        else:
            print(f"No details found for video {video_file} in JSON.")


if __name__ == "__main__":
    json_file = '../CSVs and JSONs/WLASL_v0.3.json'
    video_folder = 'Top 50 Videos/'
    output_dir = 'Top 50 Videos Processed/'
    main(json_file, video_folder, output_dir)
