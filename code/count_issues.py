import polars as pl

def count_issues(transcript, keywords):
    # Split the keywords into a list
    keyword_list = keywords.split('|')

    # Ensure transcript is a string
    transcript_str = str(transcript)
    print(transcript_str)

    # Count occurrences without "!" and with "!"
    occurrences_without_not = sum(transcript_str.count(keyword) for keyword in keyword_list)
    occurrences_with_not = sum(transcript_str.count('!' + keyword) for keyword in keyword_list)

    # Calculate the final count
    final_count = occurrences_without_not - occurrences_with_not

    return final_count

if __name__ == "__main__":
    df_tr_id = pl.read_csv("data/df_tr_testing.csv")
    keywords_data = pl.read_csv("data/important_terms.csv")
    for topic, keywords in zip(keywords_data["yt"], keywords_data["word"]):
        column_name = f"{topic}_count"
        df_tr_id_is = (
                df_tr_id.with_columns(pl.lit(999).alias(column_name))
                .with_columns(
                pl.col(column_name).map_elements(
                    lambda transcript: count_issues(transcript, keywords),
                    return_dtype=pl.Int32,
                )
            )
        )
    print(df_tr_id_is)
    print(isinstance(df_tr_id_is, pl.DataFrame))
    df_tr_id_is.write_csv("data/df_tr_id_is_testing.csv")