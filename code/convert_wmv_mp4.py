import os
import subprocess

def convert_wmv_to_mp4(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".wmv"):
            wmv_path = os.path.join(directory_path, filename)
            mp4_path = os.path.join(directory_path, filename.rsplit('.', 1)[0] + ".mp4")

            # Use FFmpeg to convert .wmv to .mp4
            subprocess.run(['ffmpeg', '-i', wmv_path, mp4_path])

            print(f'Converted: {filename} -> {filename.rsplit(".", 1)[0]}.mp4')
