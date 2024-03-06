import os
import polars as pl
from pydub import AudioSegment

def split_audio(input_dir, output_dir, segments_df = None, segments_csv = None):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if segments_csv is not None:
        # load segments file
        segments_df = pl.read_csv(segments_csv)

    # get list of files in csv
    audio_files = segments_df['file_id'].unique().to_list()

    # set padding for short segments
    pad_ms = 1000
    silence = AudioSegment.silent(duration=pad_ms)

    for file in audio_files:
        temp = segments_df.filter(pl.col('file_id') == file)
        audio = AudioSegment.from_wav(input_dir + file + ".wav")
        output_path = f"{output_dir}/{file}"
        for row in temp.rows(named=True):
            start_ms = int(row['start'] * 1000)
            start_time = row['start']
            end_ms = int(row['end'] * 1000)
            print(start_ms)
            print(end_ms)
            segment = audio[start_ms:end_ms]
            # pad short audio segments with silence to ensure proper embedding
            if row['duration'] < .5:
                segment = segment + silence
            
            segment.export(output_path + "/" + "_" + str(round(start_time, 3)) + ".wav", format="wav")


if __name__ == "__main__":
    # Specify the input directory containing WAV files, output directory, and CSV file with segments
    input_directory = "data/ad_audio_testing"
    output_directory = "data/ad_audio_split_testing"
    segments_csv_file = "data/ad_validation_initial.csv"

    # Process all WAV files in the input directory
    split_audio(input_directory, output_directory, segments_csv = segments_csv_file)
