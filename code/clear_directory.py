import os
import polars as pl

def read_rttm_file(file_path):
    # Read an RTTM file and return its content as a DataFrame
    columns = ["Type", "FileID", "Channel", "StartTime", "Duration", "SpeakerType", "Confidence", "SpeakerID", "Signal"]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [line.strip().split()[:9] for line in lines]
    df = pl.DataFrame(data, schema=columns)

    return df

def combine_rttm_files(directory_path):
    # Combine all RTTM files in a directory into a single DataFrame
    all_dataframes = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".rttm"):
            file_path = os.path.join(directory_path, filename)
            df = read_rttm_file(file_path)
            all_dataframes.append(df)

    if not all_dataframes:
        print("No .rttm files found in the specified directory.")
        return None

    combined_df = pl.concat(all_dataframes)
    return combined_df

if __name__ == "__main__":
    directory_path = "../data/ad_audio_diarized"
    # convert rttm files to single csv
    result_df = combine_rttm_files(directory_path)

    # select, rename, create columns
    df = (
    result_df
    .select(pl.col("FileID", "SpeakerID"), pl.col("StartTime", "Duration").cast(pl.Float32))
    .rename({"FileID": "file_id", "StartTime": "start", "Duration": "duration", "SpeakerID": "speaker_est"})
    .with_columns((pl.col("start") + pl.col("duration")).alias("end"))
    )

    # save result
    df.write_csv("../data/ad_validation_testing.csv")