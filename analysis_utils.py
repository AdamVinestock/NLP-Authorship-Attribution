import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from tabulate import tabulate

def extract_info_from_path(path):
    """
    Extracts dataset name, author, context policy and model from the given path.
    """
    parts = path.split('/')
    name_parts = parts[-1].split('_')
    dataset_name = name_parts[0].replace('-',' ').capitalize()
    author = name_parts[1].capitalize() + " " + name_parts[2].capitalize()
    context_policy = name_parts[3].replace('-', ' ').capitalize()
    model = name_parts[4].replace('-',' ').replace('.csv','').capitalize()
    return dataset_name, author, model, context_policy

def compute_roc_values(auth1_df, auth2_df):
    """
    input: auth1_df, auth2_df holding response values over the same LM
    :return: ROC values where labels are 1 for auth1 and 0 for auth2, threshold is perplexity value and
    TP are all author1 responses with perplexity value above threshold
    FP are all author2 responses with perplexity value above threshold
    TN are all author2 responses with perplexity value below threshold
    FN are all author1 responses with perplexity value below threshold
    """
    labels = np.concatenate([np.zeros(len(auth1_df)), np.ones(len(auth2_df))])
    responses = np.concatenate([auth1_df['response'], auth2_df['response']])

    # Handle NaN values
    nan_mask = np.isnan(responses)
    labels = labels[~nan_mask]
    responses = responses[~nan_mask]

    fpr, tpr, _ = roc_curve(labels, responses)
    roc_auc = auc(fpr, tpr)

    if roc_auc < 0.5:   # handle case where separation is below random, flip predictions
        roc_auc = 1 - roc_auc

    return fpr, tpr, roc_auc

def calc_diff(auth1_path, auth2_path):
    """
    input: paths of author1 and author2 csv's holding responses for each sentence over the same LM
    output: Standardized Mean Difference between the authors responses (author1_log-ppx - author2_log-ppx)/pooled_std
    """
    auth1_df, auth2_df = pd.read_csv(auth1_path), pd.read_csv(auth2_path)
    auth1_mean, auth2_mean = auth1_df["response"].mean(), auth2_df["response"].mean()
    auth1_std, auth2_std = auth1_df["response"].std(), auth2_df["response"].std()
    len_auth1, len_auth2 = len(auth1_df), len(auth2_df)
    pooled_std = np.sqrt(((len_auth1-1) * auth1_std**2 + (len_auth2-1) * auth2_std**2)/ (len_auth1 + len_auth2 -2))
    diff = abs((auth1_mean - auth2_mean)/pooled_std)
    return diff

def calc_js_kde(auth1_path, auth2_path, n_points=2000):
    """
    Estimate JS distance between two sets of log-ppx 'response' values
    using a kernel density estimate (KDE).

    :param auth1_path: CSV path for author1 responses
    :param auth2_path: CSV path for author2 responses
    :param bandwidth: bw_method for gaussian_kde (can tune as needed)
    :param n_points: number of points to sample across the min-max range
    :return: JS distance (float).
    """
    df1 = pd.read_csv(auth1_path)
    df2 = pd.read_csv(auth2_path)

    resp1 = df1["response"].dropna().values
    resp2 = df2["response"].dropna().values

    # fit KDE using default bandwidth (Scott's Rule)
    kde1 = gaussian_kde(resp1)
    kde2 = gaussian_kde(resp2)

    min_val = min(resp1.min(), resp2.min())
    max_val = max(resp1.max(), resp2.max())

    x_grid = np.linspace(min_val, max_val, n_points)
    pdf1 = kde1(x_grid)
    pdf2 = kde2(x_grid)

    # convert densities to probability distributions
    pdf1 /= pdf1.sum()
    pdf2 /= pdf2.sum()

    js_dist = jensenshannon(pdf1, pdf2) 
    return js_dist**2

def lm_aggregate_metrics(lm_paths):
    """
    Compute AUC, standardized mean difference (calc_diff), JS distance for two use-cases:
    1. LM separator is one of the authors.
    2. LM separator is not one of the authors.
    :param paths: List paths containing the LM repsonses values over all authors
    :return: Results for both groups for each LM.
    """
    _, _, lm_name, _ = extract_info_from_path(lm_paths[0])
    group1_aucs, group1_diffs, group1_js = [], [], [] # LM in authors
    group2_aucs, group2_diffs, group2_js = [], [], [] # LM not in authors

    for i, auth1_path in enumerate(lm_paths):       # Iterate over authors
        for j, auth2_path in enumerate(lm_paths):
            if i >= j:                              # Avoid self-comparisons or duplicate pairs
                continue

            _, author1, _, _ = extract_info_from_path(auth1_path)
            _, author2, _, _ = extract_info_from_path(auth2_path)

            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            _, _, roc_auc = compute_roc_values(auth1_df, auth2_df)
            diff = calc_diff(auth1_path, auth2_path)
            js_dist = calc_js_kde(auth1_path, auth2_path)

            # Check if the LM matches any part of the author names
            if any(keyword in author.lower() for keyword in lm_name.lower().split() for author in [author1, author2]):
                group1_aucs.append(roc_auc)
                group1_diffs.append(diff)
                group1_js.append(js_dist)
            else:
                group2_aucs.append(roc_auc)
                group2_diffs.append(diff)
                group2_js.append(js_dist)
    return group1_aucs, group1_diffs, group1_js, group2_aucs, group2_diffs, group2_js

def lm_comparison_table(group1_aucs, group1_diffs, group1_js, group2_aucs, group2_diffs, group2_js, lm_name):
    """ table preparation for per LM separator """
    avg_auc_in_authors = np.mean(group1_aucs) if group1_aucs else 0
    avg_diff_in_authors = np.mean(group1_diffs) if group1_diffs else 0
    avg_js_in_authors = np.mean(group1_js) if group1_js else 0

    avg_auc_not_in_authors = np.mean(group2_aucs) if group2_aucs else 0
    avg_diff_not_in_authors = np.mean(group2_diffs) if group2_diffs else 0
    avg_js_not_in_authors = np.mean(group2_js) if group2_js else 0
    return [lm_name, avg_auc_in_authors, avg_auc_not_in_authors, avg_diff_in_authors, avg_diff_not_in_authors, avg_js_in_authors, avg_js_not_in_authors]

def all_lms_comparison_table(group1_aucs, group1_diffs, group1_js, group2_aucs, group2_diffs, group2_js):
    """ Table preparation for aggregate over all LMs, excluding zero values """
    avg_auc_in_authors = np.mean([x for x in group1_aucs if x != 0]) if any(group1_aucs) else 0
    avg_diff_in_authors = np.mean([x for x in group1_diffs if x != 0]) if group1_diffs else 0
    avg_js_in_authors = np.mean([x for x in group1_js if x != 0]) if group1_js else 0

    avg_auc_not_in_authors = np.mean([x for x in group2_aucs if x != 0]) if group2_aucs else 0
    avg_diff_not_in_authors = np.mean([x for x in group2_diffs if x != 0]) if group2_diffs else 0
    avg_js_not_in_authors = np.mean([x for x in group2_js if x != 0]) if group2_js else 0
    return ["All LMs", avg_auc_in_authors, avg_auc_not_in_authors, avg_diff_in_authors, avg_diff_not_in_authors, avg_js_in_authors, avg_js_not_in_authors]

def run_lm_comparison(paths):
    lm_comparison_results = []
    group1_aucs_all, group1_diffs_all, group1_js_all = [], [], []
    group2_aucs_all, group2_diffs_all, group2_js_all = [], [], []
    for lm_paths in paths:
        # lm_is_author_aucs, lm_is_author_diffs, lm_not_author_aucs, lm_not_author_diffs = lm_aggregate_metrics(lm_paths)
        g1_aucs, g1_diffs, g1_js, g2_aucs, g2_diffs, g2_js = lm_aggregate_metrics(lm_paths)
        group1_aucs_all += g1_aucs
        group1_diffs_all += g1_diffs
        group1_js_all += g1_js

        group2_aucs_all += g2_aucs
        group2_diffs_all += g2_diffs
        group2_js_all += g2_js
        dataset, _, lm_name, _ = extract_info_from_path(lm_paths[0])

        lm_comparison_results.append(lm_comparison_table(g1_aucs, g1_diffs, g1_js, g2_aucs, g2_diffs, g2_js, lm_name))

    print(f"\n=== {dataset} Dataset ===")
    
    # first table (per LM separator)
    columns = ["LM Separator", 
               "Avg AUC (LM in authors)", "Avg AUC (LM not in authors)",
               "Avg Diff (LM in authors)", "Avg Diff (LM not in authors)",
               "Avg JS (LM in authors)", "Avg JS (LM not in authors)"]
    print("\nPer-LM Comparison")
    print(tabulate(lm_comparison_results, headers=columns, tablefmt="psql"))

    # second table (aggregate over all LMs)
    all_lms_results = all_lms_comparison_table(group1_aucs_all, group1_diffs_all, group1_js_all,
                                              group2_aucs_all, group2_diffs_all, group2_js_all)
    print("\nAll LMs Aggregate")
    print(tabulate([all_lms_results], headers=columns, tablefmt="psql"))
    return 

def compare_human_to_llm(paths):
    """
    Input: response paths for a specific domain
    Output: creates a table of all human vs author_x AUC values for each LM separator
    """

    print("\n=== Full Comparison Table: Human vs Generated Texts ===")
    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []

    # Iterate over each author-pair (Human vs each generated author)
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G0 = {human_label}\nG1 = {gen_label}"]

        # iterate over each LM-separator
        for lm_response_group in paths:  
            auth1_path = lm_response_group[2]     # human author responses
            auth2_path = lm_response_group[idx]   # LLM author responses
            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            _, _, roc_auc = compute_roc_values(auth1_df, auth2_df)
            row_results.append(f"{roc_auc:.4f}")
        results_table.append(row_results)
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*(len(columns))))

def compare_human_to_llm_js(paths):
    """
    Input: response paths for a specific domain
    Output: creates a table of all human vs author_x JS distances for each LM separator
    """

    print("\n=== Full Comparison Table (JS Distance): Human vs Generated Texts ===")
    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []
    # Iterate over each author-pair (Human vs each generated author)
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G0 = {human_label}\nG1 = {gen_label}"]
        # iterate over each LM-separator
        for lm_response_group in paths:  
            auth1_path = lm_response_group[2]     # human author responses
            auth2_path = lm_response_group[idx]   # LLM author responses
            js_dist = calc_js_kde(auth1_path, auth2_path)
            row_results.append(f"{js_dist:.4f}")
        results_table.append(row_results)
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*(len(columns))))

def compare_hist(auth1_path1, auth2_path1, auth1_path2, auth2_path2):
    """
    input: paths of author1 and author2 csv holding responses for each sentence
    output: histograms of author1 and author2 log-ppx values
    """
    auth1_df1 = pd.read_csv(auth1_path1)
    auth2_df1 = pd.read_csv(auth2_path1)
    auth1_df2 = pd.read_csv(auth1_path2)
    auth2_df2 = pd.read_csv(auth2_path2)

    dataset_name, author1, model1, _ = extract_info_from_path(auth1_path1)
    _, _, model2, _ = extract_info_from_path(auth1_path2)
    _, author2, _, _ = extract_info_from_path(auth2_path2)


    # Compute ROC values for each dataset
    fpr1, tpr1, roc_auc1 = compute_roc_values(auth1_df1, auth2_df1)
    fpr2, tpr2, roc_auc2 = compute_roc_values(auth1_df2, auth2_df2)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    bins = np.arange(min(auth1_df1["response"].min(), auth2_df1["response"].min(),auth1_df2["response"].min(), auth2_df2["response"].min()),
                max(auth1_df1["response"].max(), auth2_df1["response"].max(), auth1_df2["response"].max(), auth2_df2["response"].max()),
                     0.1)
    axs[0, 0].hist(auth1_df1["response"], bins=bins, alpha=0.5, label=author1)
    axs[0, 0].hist(auth2_df1["response"], bins=bins, alpha=0.5, label=author2)
    axs[0, 0].set_title(f"LM response generator - {model1}")
    axs[0, 0].set_xlabel('Log-perplexity')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].legend()

    axs[0, 1].hist(auth1_df2["response"], bins=bins, alpha=0.5, label=author1)
    axs[0, 1].hist(auth2_df2["response"], bins=bins, alpha=0.5, label=author2)
    axs[0, 1].set_title(f"LM response generator - {model2}")
    axs[0, 1].set_xlabel('Log-perplexity')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].legend()

    # Plot the ROC curves using the computed values
    axs[1, 0].plot(fpr1, tpr1, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc1:.4f})')
    axs[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1, 0].set_xlabel('False Positive Rate')
    axs[1, 0].set_ylabel('True Positive Rate')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[1, 1].plot(fpr2, tpr2, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc2:.4f})')
    axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.suptitle(f"Dataset - {dataset_name}", fontsize=16)
    plt.tight_layout()
    plt.show()