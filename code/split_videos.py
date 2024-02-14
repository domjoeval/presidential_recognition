import csv
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_file, output_folder, csv_file):
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header row

        file_name_index = header.index('file_id')
        start_time_index = header.index('start')
        end_time_index = header.index('end')

        for row in reader:
            file_name = row[file_name_index]
            start_time = float(row[start_time_index])
            end_time = float(row[end_time_index])

            input_path = f"{input_file}/{file_name}"
            output_path = f"{output_folder}/{file_name}"

            video_clip = VideoFileClip(input_path + ".mp4").subclip(start_time, end_time)
            video_clip.write_videofile(output_path + str(round(start_time, 3)) + ".mp4", codec='libx264', audio_codec='aac')
            video_clip.close()

if __name__ == "__main__":
    input_folder = "data/speech"
    output_folder = "data/speech_split"
    csv_file = "data/speech_validation.csv"

    split_video(input_folder, output_folder, csv_file)
