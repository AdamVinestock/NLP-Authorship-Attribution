
import pandas as pd
import os

### Fix for removing responses on non valid generations

def add_id_if_missing(df, id_col, force_reset_id=False):
    """
    Ensure the DataFrame has an ID column.
    If force_reset_id is True or if id_col is missing, we reset the index and assign it as the ID.
    Otherwise, we convert the existing column to string.
    """
    if force_reset_id or id_col not in df.columns:
        df = df.reset_index(drop=True)
        df[id_col] = df.index.astype(str)
    else:
        df[id_col] = df[id_col].astype(str)
    return df

def get_valid_ids(df, clean_col, dataset_id_col):
    """
    Returns the set of sample IDs (as strings) from the cleaned dataset where the cleaned text is nonempty.
    """
    valid = df[df[clean_col].fillna("").str.strip() != ""]
    return set(valid[dataset_id_col].astype(str))

def update_response_csv(cleaned_df, clean_col, responses_csv_path, output_csv_path,
                        dataset_id_col="id", response_id_col="name", force_reset_id=False):
    """
    Loads a responses CSV, removes any rows whose response_id_col is not among the valid IDs from the cleaned_df,
    prints how many rows were kept/removed, and saves the filtered responses CSV.
    For datasets without an explicit ID, if force_reset_id is True, then the DataFrame's ID column is overwritten
    with the row index.
    """
    cleaned_df = add_id_if_missing(cleaned_df, dataset_id_col, force_reset_id=force_reset_id)
    valid_ids = get_valid_ids(cleaned_df, clean_col, dataset_id_col)
    
    responses_df = pd.read_csv(responses_csv_path)
    responses_df[response_id_col] = responses_df[response_id_col].astype(str)
    
    total_rows = len(responses_df)
    responses_df_filtered = responses_df[responses_df[response_id_col].isin(valid_ids)]
    kept_rows = len(responses_df_filtered)
    removed_rows = total_rows - kept_rows
    
    print(f"Responses CSV '{responses_csv_path}': total rows = {total_rows}, kept = {kept_rows}, removed = {removed_rows}")
    responses_df_filtered.to_csv(output_csv_path, index=False)
    print(f"Saved filtered responses to '{output_csv_path}'")

def process_dataset_responses(cleaned_df, response_paths, dataset_id_col, force_reset_id=False):
    """
    Processes all response CSV files for a dataset.
    handle each dataset/author uniquely 
    """

    flattened_paths = [p for sublist in response_paths for p in sublist]
    for responses_csv in flattened_paths:
        base_name = os.path.basename(responses_csv)
        tokens = base_name.split('_')
        if tokens[1].lower() == "human":
            clean_col = "human_text"
        else:
            clean_col = tokens[1] + "_" + tokens[2]
        output_csv = responses_csv
        print(f"Processing {responses_csv} with clean_col = '{clean_col}'")
        update_response_csv(
            cleaned_df=cleaned_df,
            clean_col=clean_col,
            responses_csv_path=responses_csv,
            output_csv_path=output_csv,
            dataset_id_col=dataset_id_col,
            response_id_col="name",
            force_reset_id=force_reset_id
        )

# Paths of dataset's
wiki_df_path = "src/wiki_dataset_clean.csv"
news_df_path = "src/news_dataset_clean.csv"
abstracts_df_path = "src/abstracts_dataset_clean.csv"

# Load datasets
wiki_df = pd.read_csv(wiki_df_path)[0:1500]
news_df = pd.read_csv(news_df_path)[0:1500]
abstracts_df = pd.read_csv(abstracts_df_path)[0:1500]

wiki_paths = [
                [ 
                  "Responses/wiki_Llama3.1_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                  "Responses/wiki_Falcon_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                  "Responses/wiki_human_text_none_Meta-Llama-3.1-8B-Instruct.csv",
                  "Responses/wiki_gpt_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                  "Responses/wiki_R1_clean_none_Meta-Llama-3.1-8B-Instruct.csv"
                ],
                [ 
                  "Responses/wiki_Llama3.1_clean_none_falcon-7b.csv",
                  "Responses/wiki_Falcon_clean_none_falcon-7b.csv",
                  "Responses/wiki_human_text_none_falcon-7b.csv",
                  "Responses/wiki_gpt_clean_none_falcon-7b.csv",
                  "Responses/wiki_R1_clean_none_falcon-7b.csv"
                ],
                [ 
                  "Responses/wiki_Llama3.1_clean_none_phi-2.csv",
                  "Responses/wiki_Falcon_clean_none_phi-2.csv",
                  "Responses/wiki_human_text_none_phi-2.csv",
                  "Responses/wiki_gpt_clean_none_phi-2.csv",
                  "Responses/wiki_R1_clean_none_phi-2.csv"
                ],
                [ 
                  "Responses/wiki_Llama3.1_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                  "Responses/wiki_Falcon_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                  "Responses/wiki_human_text_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                  "Responses/wiki_gpt_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                  "Responses/wiki_R1_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv"
                ]
]

news_paths = [
                [
                "Responses/news_Llama3.1_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/news_Falcon_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/news_human_text_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/news_gpt_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/news_R1_clean_none_Meta-Llama-3.1-8B-Instruct.csv"
                ],
                [
                "Responses/news_Llama3.1_clean_none_falcon-7b.csv",
                "Responses/news_Falcon_clean_none_falcon-7b.csv",
                "Responses/news_human_text_none_falcon-7b.csv",
                "Responses/news_gpt_clean_none_falcon-7b.csv",
                "Responses/news_R1_clean_none_falcon-7b.csv"
                ],
                [
                "Responses/news_Llama3.1_clean_none_phi-2.csv",
                "Responses/news_Falcon_clean_none_phi-2.csv",
                "Responses/news_human_text_none_phi-2.csv",
                "Responses/news_gpt_clean_none_phi-2.csv",
                "Responses/news_R1_clean_none_phi-2.csv"
                ],
                [
                "Responses/news_Llama3.1_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/news_Falcon_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/news_human_text_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/news_gpt_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/news_R1_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv"
                ]
]

abstracts_paths = [
                [
                "Responses/abstracts_Llama3.1_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/abstracts_Falcon_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/abstracts_human_text_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/abstracts_gpt_clean_none_Meta-Llama-3.1-8B-Instruct.csv",
                "Responses/abstracts_R1_clean_none_Meta-Llama-3.1-8B-Instruct.csv"
                ],
                [
                "Responses/abstracts_Llama3.1_clean_none_falcon-7b.csv",
                "Responses/abstracts_Falcon_clean_none_falcon-7b.csv",
                "Responses/abstracts_human_text_none_falcon-7b.csv",
                "Responses/abstracts_gpt_clean_none_falcon-7b.csv",
                "Responses/abstracts_R1_clean_none_falcon-7b.csv"
                ],
                [
                "Responses/abstracts_Llama3.1_clean_none_phi-2.csv",
                "Responses/abstracts_Falcon_clean_none_phi-2.csv",
                "Responses/abstracts_human_text_none_phi-2.csv",
                "Responses/abstracts_gpt_clean_none_phi-2.csv",
                "Responses/abstracts_R1_clean_none_phi-2.csv"
                ],
                [
                "Responses/abstracts_Llama3.1_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/abstracts_Falcon_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/abstracts_human_text_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/abstracts_gpt_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv",
                "Responses/abstracts_R1_clean_none_DeepSeek-R1-Distill-Qwen-7B.csv"
                ]
]

print("Processing Wiki responses")
process_dataset_responses(wiki_df, wiki_paths, dataset_id_col="id", force_reset_id=False)

print("Processing News responses")
process_dataset_responses(news_df, news_paths, dataset_id_col="id", force_reset_id=False)

print("Processing Abstracts responses")
process_dataset_responses(abstracts_df, abstracts_paths, dataset_id_col="name", force_reset_id=True)