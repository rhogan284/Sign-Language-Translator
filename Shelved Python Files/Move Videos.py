import os
import csv
import shutil


def copy_videos(csv_file, source_folder, destination_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Read the CSV file
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header if exists
        for row in csv_reader:
            video_id = row[0]  # Assuming video id is in the first column
            video_filename = f"{video_id}.mp4"  # Assuming video filenames are in format "<video_id>.mp4"
            source_path = os.path.join(source_folder, video_filename)
            destination_path = os.path.join(destination_folder, video_filename)

            # Check if video file exists in source folder
            if os.path.exists(source_path):
                # Copy the video file to the destination folder
                shutil.copy2(source_path, destination_path)
                print(f"Copied {video_filename} to {destination_folder}")
            else:
                print(f"Video file {video_filename} not found in {source_folder}")


# Example usage
csv_file = "../CSVs and JSONs/top_100_glosses.csv"
source_folder = "Sign Language Videos"
destination_folder = "Top 100 Videos"
copy_videos(csv_file, source_folder, destination_folder)
