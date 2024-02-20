import os
from pydub import AudioSegment

def extract_and_convert_audio(input_path, wav_output_path, mp3_output_path):
    # Load the audio directly from the video file
    audio = AudioSegment.from_file(input_path)

    # Save the audio as a WAV file
    audio.export(wav_output_path, format="wav")

    # Save the audio as an MP3 file
    audio.export(mp3_output_path, format="mp3")

def batch_convert(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            # Build the full paths for input and output files
            input_path = os.path.join(input_directory, filename)
            base_name = os.path.splitext(filename)[0]
            wav_output_path = os.path.join(output_directory, base_name + ".wav")
            mp3_output_path = os.path.join(output_directory, base_name + ".mp3")

            # Extract and convert audio, and save as WAV and MP3
            extract_and_convert_audio(input_path, wav_output_path, mp3_output_path)
            print(f"Converted {filename} to {os.path.basename(wav_output_path)} and {os.path.basename(mp3_output_path)}")


if __name__ == "__main__":
    # Replace these paths with your input and output directories
    input_directory = "data/speech_embedding"
    output_directory = "data/speech_embedding"

    # Perform batch conversion
    batch_convert(input_directory, output_directory)
