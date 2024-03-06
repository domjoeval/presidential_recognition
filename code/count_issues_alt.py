import polars as pl

def count_issues(transcript, keywords):
    # Split the keywords into a list
    keyword_list = keywords.split('|')

    # Ensure transcript is a string
    transcript_str = str(transcript)

    # Count occurrences without "!" and with "!"
    occurrences_without_not = sum(transcript_str.count(keyword) for keyword in keyword_list)
    occurrences_with_not = sum(transcript_str.count('!' + keyword) for keyword in keyword_list)

    # Calculate the final count
    final_count = occurrences_without_not - occurrences_with_not

    return final_count

if __name__ == "__main__":
    df_tr_id = pl.read_csv("data/df_tr_testing.csv")
    keywords_data = pl.read_csv("data/important_terms.csv")

    df_tr_id_is = df_tr_id # make copy of data for adding issue counts
    del df_tr_id # delete old

    n = df_tr_id_is.shape[0] # get number of rows
    for topic, keywords in zip(keywords_data["yt"], keywords_data["word"]):
        temp_column = []
        for i in range(0, n):
            transcript = df_tr_id_is[i, 'transcript']
            issue_count = count_issues(transcript, keywords)
            temp_column.append(issue_count)
        df_tr_id_is = df_tr_id_is.with_columns(pl.Series(name = str(topic) + "_count", values = temp_column))
    print(df_tr_id_is)
    df_tr_id_is.write_csv("data/df_tr_id_is_testing.csv")