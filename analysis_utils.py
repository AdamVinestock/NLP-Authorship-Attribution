import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import jensenshannon
from tabulate import tabulate
from tqdm import tqdm
import itertools


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

def optimal_error_exponent(path_f0, path_f1, n_points=2000):
    """
    Calculate the optimal error exponent (KL Divergence) as per Chernoff–Stein lemma 
    path_f0: CSV of responses for H0 (LM matches author)
    path_f1: CSV of responses for H1 (LM doesn't match authors)
    """
    data_f0 = pd.read_csv(path_f0)["response"].dropna().values
    data_f1 = pd.read_csv(path_f1)["response"].dropna().values
    kde_f0 = gaussian_kde(data_f0)
    kde_f1 = gaussian_kde(data_f1)
    x_min, x_max = min(data_f0.min(), data_f1.min()), max(data_f0.max(), data_f1.max())
    x = np.linspace(x_min, x_max, n_points)
    pdf_f0 = kde_f0(x)
    pdf_f1 = kde_f1(x)

    # convert densities to probability distributions
    pdf_f0 /= pdf_f0.sum()
    pdf_f1 /= pdf_f1.sum()

    # numerical stability for calculating KL divergence
    pdf_f0 += 1e-10
    pdf_f1 += 1e-10
    kl_divergence = entropy(pdf_f0, pdf_f1)
    return kl_divergence

def lm_aggregate_metrics(lm_paths):
    """
    Compute AUC, standardized mean difference (calc_diff), JS distance for two use-cases:
    1. LM separator is one of the authors.
    2. LM separator is not one of the authors.
    :param paths: List paths containing the LM responses values over all authors
    :return: Results for both groups for each LM.
    """
    _, _, lm_name, _ = extract_info_from_path(lm_paths[0])
    group1_aucs, group1_diffs, group1_js, group1_oe = [], [], [], []  # LM in authors
    group2_aucs, group2_diffs, group2_js, group2_oe = [], [], [], []  # LM not in authors

    for i, auth1_path in enumerate(lm_paths):
        for j, auth2_path in enumerate(lm_paths):
            if i >= j:
                continue

            _, author1, _, _ = extract_info_from_path(auth1_path)
            _, author2, _, _ = extract_info_from_path(auth2_path)

            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            _, _, roc_auc = compute_roc_values(auth1_df, auth2_df)
            diff = calc_diff(auth1_path, auth2_path)
            js_dist = calc_js_kde(auth1_path, auth2_path)
            oe_exp = optimal_error_exponent(auth1_path, auth2_path)

            if any(keyword in author.lower() for keyword in lm_name.lower().split() for author in [author1, author2]):
                group1_aucs.append(roc_auc)
                group1_diffs.append(diff)
                group1_js.append(js_dist)
                group1_oe.append(oe_exp)
            else:
                group2_aucs.append(roc_auc)
                group2_diffs.append(diff)
                group2_js.append(js_dist)
                group2_oe.append(oe_exp)

    return group1_aucs, group1_diffs, group1_js, group1_oe, group2_aucs, group2_diffs, group2_js, group2_oe


def lm_comparison_table(group1_aucs, group1_diffs, group1_js, group1_oe, group2_aucs, group2_diffs, group2_js, group2_oe, lm_name):
    avg_auc_in_authors = np.mean(group1_aucs) if group1_aucs else 0
    avg_diff_in_authors = np.mean(group1_diffs) if group1_diffs else 0
    avg_js_in_authors = np.mean(group1_js) if group1_js else 0
    avg_oe_in_authors = np.mean(group1_oe) if group1_oe else 0

    avg_auc_not_in_authors = np.mean(group2_aucs) if group2_aucs else 0
    avg_diff_not_in_authors = np.mean(group2_diffs) if group2_diffs else 0
    avg_js_not_in_authors = np.mean(group2_js) if group2_js else 0
    avg_oe_not_in_authors = np.mean(group2_oe) if group2_oe else 0

    return [lm_name, 
            avg_auc_in_authors, avg_auc_not_in_authors,
            avg_diff_in_authors, avg_diff_not_in_authors,
            avg_js_in_authors, avg_js_not_in_authors,
            avg_oe_in_authors, avg_oe_not_in_authors]

def all_lms_comparison_table(group1_aucs, group1_diffs, group1_js, group1_oe, group2_aucs, group2_diffs, group2_js, group2_oe):
    avg_auc_in_authors = np.mean(group1_aucs) if group1_aucs else 0
    avg_diff_in_authors = np.mean(group1_diffs) if group1_diffs else 0
    avg_js_in_authors = np.mean(group1_js) if group1_js else 0
    avg_oe_in_authors = np.mean(group1_oe) if group1_oe else 0

    avg_auc_not_in_authors = np.mean(group2_aucs) if group2_aucs else 0
    avg_diff_not_in_authors = np.mean(group2_diffs) if group2_diffs else 0
    avg_js_not_in_authors = np.mean(group2_js) if group2_js else 0
    avg_oe_not_in_authors = np.mean(group2_oe) if group2_oe else 0

    return ["All LMs",
            avg_auc_in_authors, avg_auc_not_in_authors,
            avg_diff_in_authors, avg_diff_not_in_authors,
            avg_js_in_authors, avg_js_not_in_authors,
            avg_oe_in_authors, avg_oe_not_in_authors]

def run_lm_comparison(paths):
    lm_comparison_results = []
    g1_aucs_all, g1_diffs_all, g1_js_all, g1_oe_all = [], [], [], []
    g2_aucs_all, g2_diffs_all, g2_js_all, g2_oe_all = [], [], [], []

    for lm_paths in paths:
        g1_aucs, g1_diffs, g1_js, g1_oe, g2_aucs, g2_diffs, g2_js, g2_oe = lm_aggregate_metrics(lm_paths)
        g1_aucs_all += g1_aucs
        g1_diffs_all += g1_diffs
        g1_js_all += g1_js
        g1_oe_all += g1_oe

        g2_aucs_all += g2_aucs
        g2_diffs_all += g2_diffs
        g2_js_all += g2_js
        g2_oe_all += g2_oe

        dataset, _, lm_name, _ = extract_info_from_path(lm_paths[0])
        lm_comparison_results.append(lm_comparison_table(g1_aucs, g1_diffs, g1_js, g1_oe, g2_aucs, g2_diffs, g2_js, g2_oe, lm_name))

    print(f"\n=== {dataset} Dataset ===")

    columns = ["LM Separator",
               "AUC (LM in)", "AUC (LM out)",
               "Diff (LM in)", "Diff (LM out)",
               "JS (LM in)", "JS (LM out)",
               "OE (LM in)", "OE (LM out)"]

    # first table (per LM separator)
    print("\nPer-LM Comparison")
    print(tabulate(lm_comparison_results, headers=columns, tablefmt="psql"))

    # second table (aggregate over all LMs)
    all_results = all_lms_comparison_table(g1_aucs_all, g1_diffs_all, g1_js_all, g1_oe_all,
                                          g2_aucs_all, g2_diffs_all, g2_js_all, g2_oe_all)
    print("\nAll LMs Aggregate")
    print(tabulate([all_results], headers=columns, tablefmt="psql"))
    return 

def bootstrap_auc(auth1_df, auth2_df, n_bootstrap=100, ci=95):
    """
    calcs AUC between author responses, uses bootstrapping to get STD and confidence interval of calculated AUC
    """
    responses_auth1 = auth1_df['response'].values
    responses_auth2 = auth2_df['response'].values
    
    responses = np.concatenate([responses_auth1, responses_auth2])
    labels = np.concatenate([np.zeros(len(responses_auth1)), np.ones(len(responses_auth2))])

    # Handle NaN values
    nan_mask = np.isnan(responses)
    labels = labels[~nan_mask]
    responses = responses[~nan_mask]

    rng = np.random.default_rng(seed=42)
    auc_scores = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        indices = rng.choice(len(responses), size=len(responses), replace=True)
        sampled_data = responses[indices]
        sampled_labels = labels[indices]

        auc = roc_auc_score(sampled_labels, sampled_data)
        if auc < 0.5:  # flip if below random
            auc = 1 - auc
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    lower_bound = np.percentile(auc_scores, (100 - ci) / 2)
    upper_bound = np.percentile(auc_scores, 100 - (100 - ci) / 2)

    return mean_auc, std_auc, lower_bound, upper_bound



#######################################
### Methods for testing LM vs human ###
#######################################

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
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]

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

def compare_human_to_llm_ci(paths, n_bootstrap=100):
    """
    Creates a table comparing human to LLM with bootstrapped CI and STD for AUC values
    """
    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)
    ]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]
        for lm_response_group in tqdm(paths, desc=f"{gen_label} comparisons"):
            auth1_path = lm_response_group[2]   # human responses
            auth2_path = lm_response_group[idx]  # LLM responses
            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            mean_auc, std_auc, ci_lower, ci_upper = bootstrap_auc(auth1_df, auth2_df, n_bootstrap)
            row_results.append(f"{mean_auc:.3f}±{std_auc:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]")
        results_table.append(row_results)

    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    print(f"\n=== {dataset} Dataset Comparison Table with 95% CI (Human vs Generated Texts) ===")
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*len(columns)))

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
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]
        # iterate over each LM-separator
        for lm_response_group in paths:  
            auth1_path = lm_response_group[2]     # human author responses
            auth2_path = lm_response_group[idx]   # LLM author responses
            js_dist = calc_js_kde(auth1_path, auth2_path)
            row_results.append(f"{js_dist:.4f}")
        results_table.append(row_results)
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*(len(columns))))

def compare_human_to_llm_oe(paths, n_points=2000):
    """
    Similar to compare_human_to_llm_js(), but uses the Chernoff–Stein lemma exponent
    (KL divergence) via optimal_error_exponent() to compare Humans vs. LLM.

    :param paths: The list of list-of-paths structure, e.g. wiki_paths
    :param n_points: Number of discretization points for kernel density
    """
    print("\n=== Full Comparison Table (Optimal Error Exponent): Human vs Generated Texts ===")

    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)
    ]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]
        for lm_response_group in paths:  
            auth1_path = lm_response_group[2]     # human
            auth2_path = lm_response_group[idx]   # LLM
            # compute OE
            oe_val = optimal_error_exponent(auth1_path, auth2_path, n_points=n_points)
            row_results.append(f"{oe_val:.4f}")
        results_table.append(row_results)

    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*(len(columns))))


############################################################
### Methods for comparing lm vs other Author separations ###
############################################################

def compare_author_vs_others(paths, metric='js'):
    """
    Creates a table comparing the average metric of each author against all other authors using a selected metric.

    :param paths: Nested list of response paths organized by LM separators
    :param metric: Metric to calculate ('js', 'auc', 'optimal_error')
    """
    metrics_map = {
        'js': calc_js_kde,
        'auc': compute_roc_values,
        'optimal_error': optimal_error_exponent
    }
    author_labels = ['Llama3.1', 'Falcon', 'Human', 'GPT', 'R1']
    separator_labels = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    columns = ["Author vs Others"] + separator_labels
    results_table = []
    for author_idx, author_label in enumerate(author_labels):
        row_results = [f"{author_label} vs Others"]
        for separator_group in paths:
            metrics = []
            for other_idx in range(len(author_labels)):
                if other_idx != author_idx:
                    author_path = separator_group[author_idx]
                    other_author_path = separator_group[other_idx]
                    if metric == 'auc':
                        auth1_df = pd.read_csv(author_path)
                        auth2_df = pd.read_csv(other_author_path)
                        _, _, metric_value = metrics_map[metric](auth1_df, auth2_df)
                    else:
                        metric_value = metrics_map[metric](author_path, other_author_path)
                    metrics.append(metric_value)
            avg_metric = sum(metrics) / len(metrics)
            row_results.append(f"{avg_metric:.4f}")
        results_table.append(row_results)
    results_df = pd.DataFrame(results_table, columns=columns)
    print(f"\n=== Average Author Separation using {metric.upper()} ===")
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*(len(columns))))
    # print(results_df.to_markdown(index=False))

############################################################################
### Methods for printing responses histogram comparisons/saving tables  ####
############################################################################


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

def save_ascii_table_as_png(table_str, filename="table.png", 
                            fig_width=16, font_family="monospace", 
                            font_size=12, line_height=0.45):
    """
    Saves an ASCII table string (e.g. from tabulate) as a PNG image 
    with a white background in a monospaced font.
    """
    lines = table_str.count("\n") + 1
    fig_height = line_height * lines

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.text(0.0, 1.0, table_str, 
            fontfamily=font_family, 
            fontsize=font_size, 
            va="top")

    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def run_lm_comparison_save_img(paths, save_prefix=None):
    lm_comparison_results = []
    group1_aucs_all, group1_diffs_all, group1_js_all, group1_oe_all = [], [], [], []
    group2_aucs_all, group2_diffs_all, group2_js_all, group2_oe_all = [], [], [], []
    for lm_paths in paths:
        g1_aucs, g1_diffs, g1_js, g1_oe,  g2_aucs, g2_diffs, g2_js, g2_oe = lm_aggregate_metrics(lm_paths)
        group1_aucs_all += g1_aucs
        group1_diffs_all += g1_diffs
        group1_js_all += g1_js
        group1_oe_all += g1_oe

        group2_aucs_all += g2_aucs
        group2_diffs_all += g2_diffs
        group2_js_all += g2_js
        group2_oe_all += g2_oe

        dataset, _, lm_name, _ = extract_info_from_path(lm_paths[0])
        lm_comparison_results.append(
            lm_comparison_table(g1_aucs, g1_diffs, g1_js, g1_oe,
                                g2_aucs, g2_diffs, g2_js, g2_oe,
                                lm_name)
        )
    print(f"\n=== {dataset} Dataset ===")
    
    # first table (per LM separator)
    columns = ["LM Separator",
            "AUC (LM in)", "AUC (LM out)",
            "Diff (LM in)", "Diff (LM out)",
            "JS (LM in)", "JS (LM out)",
            "OE (LM in)", "OE (LM out)"]

    print("\nPer-LM Comparison")
    per_lm_table_str = tabulate(lm_comparison_results, headers=columns, tablefmt="psql")
    print(per_lm_table_str)

    # second table (aggregate over all LMs)
    all_lms_results = all_lms_comparison_table(
        group1_aucs_all, group1_diffs_all, group1_js_all, group1_oe_all,
        group2_aucs_all, group2_diffs_all, group2_js_all, group2_oe_all
    )
    print("\nAll LMs Aggregate")
    all_lms_table_str = tabulate([all_lms_results], headers=columns, tablefmt="psql")
    print(all_lms_table_str)
    if save_prefix is not None:
        save_ascii_table_as_png(
            per_lm_table_str,
            filename=f"images/{save_prefix}_per_lm.png",
            fig_width=20  
        )
        save_ascii_table_as_png(
            all_lms_table_str,
            filename=f"images/{save_prefix}_agg.png",
            fig_width=16
        )
    return

def compare_separators(dataset_paths, author1, author2):
    separator_names = ['Llama3.1', 'Falcon', 'Phi-2', 'DeepSeek-R1']
    num_separators = len(separator_names)

    fig, axs = plt.subplots(2, num_separators, figsize=(5 * num_separators, 10))

    for idx, separator in enumerate(separator_names):
        separator_paths = dataset_paths[idx]
        dataset_name, _, model_name, _ = extract_info_from_path(separator_paths[0])
        auth1_path, auth2_path = None, None
        for path in separator_paths:
            _, author, _, _ = extract_info_from_path(path)
            if author1.lower() in author.lower():
                auth1_path = path
            elif author2.lower() in author.lower():
                auth2_path = path

        if auth1_path is None or auth2_path is None:
            raise ValueError(f"Paths for authors '{author1}' or '{author2}' not found in {separator}.")

        auth1_df = pd.read_csv(auth1_path)
        auth2_df = pd.read_csv(auth2_path)
        fpr, tpr, roc_auc = compute_roc_values(auth1_df, auth2_df)

        # plot histograms
        bins = np.arange(min(auth1_df["response"].min(), auth2_df["response"].min()),
                         max(auth1_df["response"].max(), auth2_df["response"].max()), 0.1)

        axs[0, idx].hist(auth1_df["response"], bins=bins, alpha=0.5, label=author1)
        axs[0, idx].hist(auth2_df["response"], bins=bins, alpha=0.5, label=author2)
        axs[0, idx].set_title(f"{separator} separator")
        axs[0, idx].set_xlabel('Log-perplexity')
        axs[0, idx].set_ylabel('Frequency')
        axs[0, idx].legend()

        # plot ROC curve
        axs[1, idx].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.4f})')
        axs[1, idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[1, idx].set_xlabel('False Positive Rate')
        axs[1, idx].set_ylabel('True Positive Rate')
        axs[1, idx].legend(loc='lower right')
        axs[1, idx].grid(True, linestyle='--', linewidth=0.5)

    plt.suptitle(f"Comparison of Separators for {author1} vs {author2} ({dataset_name} dataset)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def compare_separators_with_kl(dataset_paths, author1, author2):
    separator_names = ['Llama3.1', 'Falcon', 'Phi-2', 'DeepSeek-R1']
    num_separators = len(separator_names)
    fig, axs = plt.subplots(2, num_separators, figsize=(5 * num_separators, 10))
    kl_values = []
    for idx, separator in enumerate(separator_names):
        separator_paths = dataset_paths[idx]
        dataset_name, _, model_name, _ = extract_info_from_path(separator_paths[0])

        auth1_path, auth2_path = None, None
        for path in separator_paths:
            _, author, _, _ = extract_info_from_path(path)
            if author1.lower() in author.lower():
                auth1_path = path
            elif author2.lower() in author.lower():
                auth2_path = path

        if auth1_path is None or auth2_path is None:
            raise ValueError(f"Paths for authors '{author1}' or '{author2}' not found in {separator}.")

        auth1_df = pd.read_csv(auth1_path)
        auth2_df = pd.read_csv(auth2_path)

        bins = np.arange(min(auth1_df["response"].min(), auth2_df["response"].min()),
                         max(auth1_df["response"].max(), auth2_df["response"].max()), 0.1)

        # Histograms
        axs[0, idx].hist(auth1_df["response"], bins=bins, alpha=0.5, label=author1)
        axs[0, idx].hist(auth2_df["response"], bins=bins, alpha=0.5, label=author2)
        axs[0, idx].set_title(f"{separator} separator")
        axs[0, idx].set_xlabel('Log-perplexity')
        axs[0, idx].set_ylabel('Frequency')
        axs[0, idx].legend()

        # KL Divergence plot
        kl_divergence = optimal_error_exponent_viz(
            auth1_path, auth2_path, axs[1, idx], author1, author2, f"{separator} separator"
        )
        kl_values.append(kl_divergence)

    plt.suptitle(f"Comparison using Optimal Error Exponent (KL Divergence)\n{author1} vs {author2} ({dataset_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # Summary bar plot for KL divergence values
    plt.figure(figsize=(10, 5))
    plt.bar(separator_names, kl_values, color='skyblue')
    plt.ylabel('KL Divergence (Optimal Error Exponent)')
    plt.title('Comparison of KL Divergence Across Separators')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


############################################################################
### Clean Results Functions (AUC only with standard errors)            ####
############################################################################

def bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap=100):
    """
    Calculate AUC with bootstrapped standard error (no confidence intervals)
    Returns mean_auc, std_error
    """
    responses_auth1 = auth1_df['response'].values
    responses_auth2 = auth2_df['response'].values
    
    responses = np.concatenate([responses_auth1, responses_auth2])
    labels = np.concatenate([np.zeros(len(responses_auth1)), np.ones(len(responses_auth2))])

    # Handle NaN values
    nan_mask = np.isnan(responses)
    labels = labels[~nan_mask]
    responses = responses[~nan_mask]

    rng = np.random.default_rng(seed=42)
    auc_scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(responses), size=len(responses), replace=True)
        sampled_data = responses[indices]
        sampled_labels = labels[indices]

        auc = roc_auc_score(sampled_labels, sampled_data)
        if auc < 0.5:  # flip if below random
            auc = 1 - auc
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    std_error = np.std(auc_scores)
    return mean_auc, std_error

def lm_aggregate_metrics_clean(lm_paths, n_bootstrap=100):
    """
    Compute AUC with standard errors for two use-cases:
    1. LM separator is one of the authors.
    2. LM separator is not one of the authors.
    Returns: group1_aucs, group1_stds, group2_aucs, group2_stds
    """
    _, _, lm_name, _ = extract_info_from_path(lm_paths[0])
    group1_aucs, group1_stds = [], []  # LM in authors
    group2_aucs, group2_stds = [], []  # LM not in authors

    for i, auth1_path in enumerate(lm_paths):
        for j, auth2_path in enumerate(lm_paths):
            if i >= j:
                continue

            _, author1, _, _ = extract_info_from_path(auth1_path)
            _, author2, _, _ = extract_info_from_path(auth2_path)

            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)

            if any(keyword in author.lower() for keyword in lm_name.lower().split() for author in [author1, author2]):
                group1_aucs.append(mean_auc)
                group1_stds.append(std_error)
            else:
                group2_aucs.append(mean_auc)
                group2_stds.append(std_error)

    return group1_aucs, group1_stds, group2_aucs, group2_stds

def lm_comparison_table_clean(group1_aucs, group1_stds, group2_aucs, group2_stds, lm_name):
    """Create clean table row with AUC and standard errors in brackets"""
    if group1_aucs:
        avg_auc_in = np.mean(group1_aucs)
        avg_std_in = np.sqrt(np.mean(np.array(group1_stds)**2))  # pooled standard error
        auc_in_str = f"{avg_auc_in:.3f} ({avg_std_in:.3f})"
    else:
        auc_in_str = "N/A"
    
    if group2_aucs:
        avg_auc_out = np.mean(group2_aucs)
        avg_std_out = np.sqrt(np.mean(np.array(group2_stds)**2))  # pooled standard error
        auc_out_str = f"{avg_auc_out:.3f} ({avg_std_out:.3f})"
    else:
        auc_out_str = "N/A"

    return [lm_name, auc_in_str, auc_out_str]

def run_lm_comparison_clean(paths, n_bootstrap=100):
    """
    Generate clean comparison table with only AUC and standard errors
    """
    lm_comparison_results = []
    g1_aucs_all, g1_stds_all = [], []
    g2_aucs_all, g2_stds_all = [], []

    for lm_paths in paths:
        g1_aucs, g1_stds, g2_aucs, g2_stds = lm_aggregate_metrics_clean(lm_paths, n_bootstrap)
        g1_aucs_all.extend(g1_aucs)
        g1_stds_all.extend(g1_stds)
        g2_aucs_all.extend(g2_aucs)
        g2_stds_all.extend(g2_stds)

        dataset, _, lm_name, _ = extract_info_from_path(lm_paths[0])
        lm_comparison_results.append(lm_comparison_table_clean(g1_aucs, g1_stds, g2_aucs, g2_stds, lm_name))

    print(f"\n=== {dataset} Dataset - Clean Results (AUC with Standard Errors) ===")

    columns = ["LM Separator", "AUC (LM ∈ authors)", "AUC (LM ∉ authors)"]

    # Per-LM table
    print("\nPer-LM Results:")
    print(tabulate(lm_comparison_results, headers=columns, tablefmt="grid"))

    # Aggregate results
    if g1_aucs_all:
        agg_auc_in = np.mean(g1_aucs_all)
        agg_std_in = np.sqrt(np.mean(np.array(g1_stds_all)**2))
        agg_in_str = f"{agg_auc_in:.3f} ({agg_std_in:.3f})"
    else:
        agg_in_str = "N/A"
    
    if g2_aucs_all:
        agg_auc_out = np.mean(g2_aucs_all)
        agg_std_out = np.sqrt(np.mean(np.array(g2_stds_all)**2))
        agg_out_str = f"{agg_auc_out:.3f} ({agg_std_out:.3f})"
    else:
        agg_out_str = "N/A"

    aggregate_row = ["All LMs", agg_in_str, agg_out_str]
    
    print("\nAggregate Results:")
    print(tabulate([aggregate_row], headers=columns, tablefmt="grid"))
    
    return lm_comparison_results, aggregate_row

def compare_human_to_llm_clean(paths, n_bootstrap=100):
    """
    Creates clean table comparing human to LLM with AUC and standard errors only
    """
    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)
    ]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []
    
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]
        for lm_response_group in paths:
            auth1_path = lm_response_group[2]   # human responses
            auth2_path = lm_response_group[idx]  # LLM responses
            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
            row_results.append(f"{mean_auc:.3f}\n({std_error:.3f})")
        results_table.append(row_results)

    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    print(f"\n=== {dataset} Dataset - Clean Human vs Generated Results (AUC with Standard Errors) ===")
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*len(columns)))
    
    return results_table


############################################################################
### Appendix Functions (JS and OE detailed results)                    ####
############################################################################

def compare_human_to_llm_appendix_js(paths):
    """
    Creates detailed JS distance table for appendix
    """
    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)
    ]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []
    
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]
        for lm_response_group in paths:
            auth1_path = lm_response_group[2]     # human author responses
            auth2_path = lm_response_group[idx]   # LLM author responses
            js_dist = calc_js_kde(auth1_path, auth2_path)
            row_results.append(f"{js_dist:.4f}")
        results_table.append(row_results)
    
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    print(f"\n=== {dataset} Dataset - Appendix: JS Distance Results ===")
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*len(columns)))
    
    return results_table

def compare_human_to_llm_appendix_oe(paths):
    """
    Creates detailed OE (KL divergence) table for appendix
    """
    comparisons = [
        ("Humans", "Llama", 0),
        ("Humans", "Falcon", 1),
        ("Humans", "GPT", 3),
        ("Humans", "R1", 4)
    ]
    columns = ["", "Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    results_table = []
    
    for human_label, gen_label, idx in comparisons:
        row_results = [f"G1 = {human_label}\nG0 = {gen_label}"]
        for lm_response_group in paths:
            auth1_path = lm_response_group[2]     # human
            auth2_path = lm_response_group[idx]   # LLM
            oe_val = optimal_error_exponent(auth1_path, auth2_path)
            row_results.append(f"{oe_val:.4f}")
        results_table.append(row_results)

    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    print(f"\n=== {dataset} Dataset - Appendix: Optimal Error Exponent Results ===")
    print(tabulate(results_table, headers=columns, tablefmt="grid", colalign=("center",)*len(columns)))
    
    return results_table

def run_lm_comparison_appendix_js_oe(paths):
    """
    Generate detailed JS and OE tables for appendix
    """
    lm_js_results = []
    lm_oe_results = []
    
    for lm_paths in paths:
        g1_aucs, g1_diffs, g1_js, g1_oe, g2_aucs, g2_diffs, g2_js, g2_oe = lm_aggregate_metrics(lm_paths)
        
        dataset, _, lm_name, _ = extract_info_from_path(lm_paths[0])
        
        # JS results
        avg_js_in = np.mean(g1_js) if g1_js else np.nan
        avg_js_out = np.mean(g2_js) if g2_js else np.nan
        lm_js_results.append([lm_name, f"{avg_js_in:.4f}", f"{avg_js_out:.4f}"])
        
        # OE results
        avg_oe_in = np.mean(g1_oe) if g1_oe else np.nan
        avg_oe_out = np.mean(g2_oe) if g2_oe else np.nan
        lm_oe_results.append([lm_name, f"{avg_oe_in:.4f}", f"{avg_oe_out:.4f}"])

    print(f"\n=== {dataset} Dataset - Appendix: Detailed JS Distance Results ===")
    js_columns = ["LM Separator", "JS (LM ∈ authors)", "JS (LM ∉ authors)"]
    print(tabulate(lm_js_results, headers=js_columns, tablefmt="grid"))

    print(f"\n=== {dataset} Dataset - Appendix: Detailed OE Results ===")
    oe_columns = ["LM Separator", "OE (LM ∈ authors)", "OE (LM ∉ authors)"]
    print(tabulate(lm_oe_results, headers=oe_columns, tablefmt="grid"))
    
    return lm_js_results, lm_oe_results

def optimal_error_exponent_viz(path_f0, path_f1, ax, author1, author2, title, n_points=2000):
    data_f0 = pd.read_csv(path_f0)["response"].dropna().values
    data_f1 = pd.read_csv(path_f1)["response"].dropna().values
    kde_f0 = gaussian_kde(data_f0)
    kde_f1 = gaussian_kde(data_f1)
    x_min, x_max = min(data_f0.min(), data_f1.min()), max(data_f0.max(), data_f1.max())
    x = np.linspace(x_min, x_max, n_points)
    pdf_f0 = kde_f0(x)
    pdf_f1 = kde_f1(x)

    # normalize PDFs
    pdf_f0 /= pdf_f0.sum()
    pdf_f1 /= pdf_f1.sum()

    # calc KL Divergence
    pdf_f0 += 1e-10
    pdf_f1 += 1e-10
    kl_divergence = entropy(pdf_f0, pdf_f1)

    # calc mean and std
    mean_f0, std_f0 = data_f0.mean(), data_f0.std()
    mean_f1, std_f1 = data_f1.mean(), data_f1.std()

    # plot
    ax.plot(x, pdf_f0, label=f"{author1} (mean={mean_f0:.2f}, std={std_f0:.2f})")
    ax.plot(x, pdf_f1, label=f"{author2} (mean={mean_f1:.2f}, std={std_f1:.2f})")
    ax.fill_between(x, pdf_f0, pdf_f1, color='grey', alpha=0.2)
    ax.set_title(f"{title}\nOptimal Error Exp. (KL Divergence): {kl_divergence:.4f}")
    ax.set_xlabel("Log-Perplexity")
    ax.set_ylabel("Density")
    ax.legend()
    return kl_divergence

def compare_hist_oe(auth1_path1, auth2_path1, auth1_path2, auth2_path2):
    auth1_df1 = pd.read_csv(auth1_path1)
    auth2_df1 = pd.read_csv(auth2_path1)
    auth1_df2 = pd.read_csv(auth1_path2)
    auth2_df2 = pd.read_csv(auth2_path2)

    dataset_name, author1, model1, _ = extract_info_from_path(auth1_path1)
    _, _, model2, _ = extract_info_from_path(auth1_path2)
    _, author2, _, _ = extract_info_from_path(auth2_path2)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    bins = np.arange(min(auth1_df1["response"].min(), auth2_df1["response"].min(),
                          auth1_df2["response"].min(), auth2_df2["response"].min()),
                     max(auth1_df1["response"].max(), auth2_df1["response"].max(),
                         auth1_df2["response"].max(), auth2_df2["response"].max()), 0.1)

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

    # Bottom plots: Optimal Error Exponent visualizations
    optimal_error_exponent_viz(auth1_path1, auth2_path1, axs[1, 0], author1, author2, f"{model1} Optimal Error Exponent")
    optimal_error_exponent_viz(auth1_path2, auth2_path2, axs[1, 1], author1, author2, f"{model2} Optimal Error Exponent")

    plt.suptitle(f"Dataset - {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  HEAT‑MAP PLOTS
# ──────────────────────────────────────────────────────────────────────────────

# 1.  Separating 2 authors   (detector ∈ authors   vs   detector ∉ authors)

def run_lm_comparison_heatmap(paths, save_prefix=None):
    """
    Prints/returns a table and heatmaps for each metric, with 'aggregate' row (phi-2 excluded).
    Prints domain, and number of pairs per row.
    """
    # Extract domain name for title
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    print(f"\n=== {dataset} Domain ===")

    # Collect rows, keep track of phi-2 row index
    all_rows = []
    phi2_index = None
    for i, lm_paths in enumerate(paths):
        _, _, lm_name, _ = extract_info_from_path(lm_paths[0])
        (g1_aucs, _, g1_js, g1_oe,
         g2_aucs, _, g2_js, g2_oe) = lm_aggregate_metrics(lm_paths)
        auc_in  = np.mean(g1_aucs) if g1_aucs else np.nan
        auc_out = np.mean(g2_aucs) if g2_aucs else np.nan
        js_in   = np.mean(g1_js) if g1_js else np.nan
        js_out  = np.mean(g2_js) if g2_js else np.nan
        oe_in   = np.mean(g1_oe) if g1_oe else np.nan
        oe_out  = np.mean(g2_oe) if g2_oe else np.nan
        n_in, n_out = len(g1_aucs), len(g2_aucs)
        all_rows.append([
            lm_name, auc_in, auc_out, js_in, js_out, oe_in, oe_out, n_in, n_out
        ])
        if "phi" in lm_name.lower():
            phi2_index = i

    # Reorder so phi-2 is last
    if phi2_index is not None:
        phi2_row = all_rows.pop(phi2_index)
        all_rows.append(phi2_row)

    # Aggregate row (exclude phi-2 from means)
    rows_for_agg = all_rows[:-1] if phi2_index is not None else all_rows
    agg = ["Aggregate"]
    for col in range(1, 7):
        col_vals = [row[col] for row in rows_for_agg if not np.isnan(row[col])]
        agg.append(np.mean(col_vals) if col_vals else np.nan)
    # Pair counts for aggregate (sums)
    agg.append(int(np.nansum([row[7] for row in rows_for_agg])))
    agg.append(int(np.nansum([row[8] for row in rows_for_agg])))
    all_rows.append(agg)

    # DataFrame for main results
    columns = [
        "LM Separator", "AUC (∈authors)", "AUC (∉authors)",
        "JS (∈authors)", "JS (∉authors)",
        "OE (∈authors)", "OE (∉authors)",
        "#Pairs (∈authors)", "#Pairs (∉authors)"
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    print("\nMain Results Table (Aggregate excludes phi-2):")
    print(df.to_markdown(index=False, floatfmt=".3f"))

    # Plot heatmaps for each metric (aggregate last row, phi-2 before that)
    fig, axs = plt.subplots(1, 3, figsize=(15, len(df)*0.6+2))
    for idx, (col_in, col_out, cmap, title) in enumerate([
        ("AUC (∈authors)", "AUC (∉authors)", "Blues", "AUC"),
        ("JS (∈authors)", "JS (∉authors)", "Reds", "JS"),
        ("OE (∈authors)", "OE (∉authors)", "Purples", "OE")
    ]):
        vals = np.array([
            df[col_in].astype(float).values,
            df[col_out].astype(float).values
        ]).T
        sns.heatmap(vals, annot=True, fmt=".3f", cmap=cmap,
                    yticklabels=df["LM Separator"],
                    xticklabels=["∈authors", "∉authors"], ax=axs[idx])
        axs[idx].set_title(title)
    plt.suptitle(f"{dataset} Domain – Separation Metrics (Aggregate excludes phi-2)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_prefix:
        plt.savefig(f"{save_prefix}_all_metrics_heatmap.png", dpi=180)
    plt.show()

    return df


################################################################################
###                    PLOTTING METHODS FOR FINAL PAPER                     ###
################################################################################



# ==============================================================================
# EXPERIMENT 1: LM Separator Analysis
# ==============================================================================

def run_lm_comparison_heatmap_clean(paths, n_bootstrap=100, save_prefix=None):
    """
    Publication-ready heatmap showing AUC with standard errors for LM separator analysis.
    Shows when LM separator is one of the authors vs. not one of the authors.
    """
    # Extract domain name for title
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Collect data with standard errors and pair counts
    all_rows = []
    phi2_index = None
    
    for i, lm_paths in enumerate(paths):
        _, _, lm_name, _ = extract_info_from_path(lm_paths[0])
        g1_aucs, g1_stds, g2_aucs, g2_stds = lm_aggregate_metrics_clean(lm_paths, n_bootstrap)
        
        # Calculate means and pooled standard errors
        if g1_aucs:
            auc_in = np.mean(g1_aucs)
            std_in = np.sqrt(np.mean(np.array(g1_stds)**2))
            n_in = len(g1_aucs)
        else:
            auc_in, std_in, n_in = np.nan, np.nan, 0
            
        if g2_aucs:
            auc_out = np.mean(g2_aucs)
            std_out = np.sqrt(np.mean(np.array(g2_stds)**2))
            n_out = len(g2_aucs)
        else:
            auc_out, std_out, n_out = np.nan, np.nan, 0
        
        all_rows.append([lm_name, auc_in, std_in, auc_out, std_out, n_in, n_out])
        if "phi" in lm_name.lower():
            phi2_index = i

    # Reorder so phi-2 is last
    if phi2_index is not None:
        phi2_row = all_rows.pop(phi2_index)
        all_rows.append(phi2_row)

    # Aggregate row (exclude phi-2 from means)
    rows_for_agg = all_rows[:-1] if phi2_index is not None else all_rows
    
    # Calculate aggregate values
    agg_auc_in_vals = [row[1] for row in rows_for_agg if not np.isnan(row[1])]
    agg_std_in_vals = [row[2] for row in rows_for_agg if not np.isnan(row[2])]
    agg_auc_out_vals = [row[3] for row in rows_for_agg if not np.isnan(row[3])]
    agg_std_out_vals = [row[4] for row in rows_for_agg if not np.isnan(row[4])]
    agg_n_in = sum([row[5] for row in rows_for_agg])
    agg_n_out = sum([row[6] for row in rows_for_agg])
    
    agg_auc_in = np.mean(agg_auc_in_vals) if agg_auc_in_vals else np.nan
    agg_std_in = np.sqrt(np.mean(np.array(agg_std_in_vals)**2)) if agg_std_in_vals else np.nan
    agg_auc_out = np.mean(agg_auc_out_vals) if agg_auc_out_vals else np.nan
    agg_std_out = np.sqrt(np.mean(np.array(agg_std_out_vals)**2)) if agg_std_out_vals else np.nan
    
    all_rows.append(["Aggregate", agg_auc_in, agg_std_in, agg_auc_out, agg_std_out, agg_n_in, agg_n_out])

    # Create DataFrame
    df = pd.DataFrame(all_rows, columns=["LM Separator", "AUC_in", "SE_in", "AUC_out", "SE_out", "N_in", "N_out"])
    
    # Print clean table
    print(f"\n=== {dataset} Dataset - Experiment 1: LM Separator Analysis ===")
    print("AUC Performance: LM ∈ Authors vs LM ∉ Authors\n")
    
    table_rows = []
    for _, row in df.iterrows():
        lm_name = row["LM Separator"]
        if not np.isnan(row["AUC_in"]):
            auc_in_str = f"{row['AUC_in']:.3f} ({row['SE_in']:.3f})"
            n_in_str = f"{int(row['N_in'])}"
        else:
            auc_in_str = "N/A"
            n_in_str = "0"
        auc_out_str = f"{row['AUC_out']:.3f} ({row['SE_out']:.3f})"
        n_out_str = f"{int(row['N_out'])}"
        table_rows.append([lm_name, auc_in_str, auc_out_str, n_in_str, n_out_str])
    
    print(tabulate(table_rows, 
                   headers=["LM Separator", "AUC (LM ∈ authors)", "AUC (LM ∉ authors)", 
                           "#Pairs (∈authors)", "#Pairs (∉authors)"], 
                   tablefmt="grid"))

    fig, ax = plt.subplots(1, 1, figsize=(8, len(df)*0.8))
    
    # Prepare data for heatmap (AUC values only)
    heatmap_data = np.array([
        df["AUC_in"].values,
        df["AUC_out"].values
    ]).T
    
    # Create annotations with AUC and standard error
    annotations = []
    for i, (_, row) in enumerate(df.iterrows()):
        if not np.isnan(row["AUC_in"]):
            ann_in = f"{row['AUC_in']:.3f}\n±{row['SE_in']:.3f}"
        else:
            ann_in = "N/A"
        ann_out = f"{row['AUC_out']:.3f}\n±{row['SE_out']:.3f}"
        annotations.append([ann_in, ann_out])
    
    # Create heatmap with custom colormap optimized for 0.6-0.7 range
    cmap = plt.cm.RdYlGn 
    
    # Mask NaN values
    mask = np.isnan(heatmap_data)
    
    # Set AUC range from 0.55 to 0.8 for better visualization of typical values
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0.55, vmax=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC Score', rotation=270, labelpad=20, fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(2))
    ax.set_xticklabels(['LM ∈ authors', 'LM ∉ authors'], fontsize=11)
    ax.set_yticks(range(len(df)))
    
    # Clean up LM names for display
    clean_names = []
    for name in df["LM Separator"]:
        if name == "Aggregate":
            clean_names.append("Aggregate")
        elif "llama" in name.lower():
            clean_names.append("Llama-3.1-8B")
        elif "falcon" in name.lower():
            clean_names.append("Falcon-7B")
        elif "phi" in name.lower():
            clean_names.append("Phi-2")
        elif "deepseek" in name.lower():
            clean_names.append("DeepSeek-R1")
        else:
            clean_names.append(name)
    
    ax.set_yticklabels(clean_names, fontsize=11)
    
    for i in range(len(df)):
        for j in range(2):
            if not mask[i, j]:
                ax.text(j, i, annotations[i][j], 
                       ha='center', va='center', 
                       fontsize=10, fontweight='bold',
                       color='black')  # Always black for readability
            else:
                ax.text(j, i, annotations[i][j], 
                       ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='gray')
    
    # Styling
    title_text = f'{dataset} Dataset: LM Separator Performance\n(AUC ± Standard Error)'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add informational text below the plot
    info_text = "Aggregate excludes Phi-2"
    
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10, 
                style='italic', wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_experiment1_auc_heatmap.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    return df, table_rows


# ==============================================================================
# EXPERIMENT 2: Author-Specific Separation Performance
# Clean and verbose versions with professional formatting
# ==============================================================================

def compare_author_vs_others_clean(paths, n_bootstrap=100, save_prefix=None):
    """
    Clean publication-ready table with heatmap for Experiment 2: Author-Specific Separation Performance.
    Shows how well each text generator can be distinguished from others using different perplexity evaluators (P).
    Best values are shown in BOLD.
    
    :param paths: Nested list of response paths organized by LM separators
    :param n_bootstrap: Number of bootstrap iterations for standard error calculation
    :param save_prefix: Optional prefix for saving heatmap image
    :return: Results table, DataFrame, and numeric values DataFrame
    """
    # Reorder authors: models used as both P and generators first (Llama3.1, Falcon, R1), then others (Human, GPT)
    author_order = [0, 1, 4, 2, 3]  # Llama3.1, Falcon, R1, Human, GPT
    author_labels_full = ['Llama3.1', 'Falcon', 'Human', 'GPT', 'R1']
    author_labels = [author_labels_full[i] for i in author_order]
    
    # Short labels for table columns
    separator_labels_short = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_labels_full = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    columns = ["Text Generator"] + separator_labels_short
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Store numeric values for heatmap and ranking
    auc_matrix = []
    std_matrix = []
    results_table = []
    
    for author_idx in author_order:
        author_label = author_labels_full[author_idx]
        row_aucs = []
        row_stds = []
        
        for separator_group in paths:
            aucs = []
            stds = []
            
            # Compare this author against all others
            for other_idx in range(len(author_labels_full)):
                if other_idx != author_idx:
                    author_path = separator_group[author_idx]
                    other_author_path = separator_group[other_idx]
                    auth1_df = pd.read_csv(author_path)
                    auth2_df = pd.read_csv(other_author_path)
                    mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
                    aucs.append(mean_auc)
                    stds.append(std_error)
            
            # Calculate average AUC and pooled standard error
            avg_auc = np.mean(aucs)
            avg_std = np.sqrt(np.mean(np.array(stds)**2))
            row_aucs.append(avg_auc)
            row_stds.append(avg_std)
        
        auc_matrix.append(row_aucs)
        std_matrix.append(row_stds)
    
    # Create table (clean, without formatting markers - visual emphasis is in heatmap)
    for i, author_idx in enumerate(author_order):
        author_label = author_labels_full[author_idx]
        
        # Grey out Human and GPT (they're not used as separators)
        if author_label in ['Human', 'GPT']:
            row_results = [f"{author_label} †"]
        else:
            row_results = [author_label]
        
        # Just show values without formatting markers (heatmap provides visual emphasis)
        for j, (auc_val, std_val) in enumerate(zip(auc_matrix[i], std_matrix[i])):
            formatted = f"{auc_val:.3f} ({std_val:.3f})"
            row_results.append(formatted)
        
        results_table.append(row_results)
    
    # Print table (LaTeX-style with horizontal lines only)
    print(f"\n=== Experiment 2: Author-Specific Separation Performance ({dataset} Dataset) ===")
    print(f"Y-axis: Text generator to be distinguished from all others")
    print(f"X-axis: Perplexity evaluator model (P) used for separation")
    print(f"Values: AUC ± Standard Error (best values highlighted in heatmap)")
    print(f"† = Not used as separator (informative only)\n")
    print(tabulate(results_table, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    # Create heatmap with slimmer rows
    fig, ax = plt.subplots(1, 1, figsize=(10, len(author_labels)*0.6 + 1.5))
    
    auc_array = np.array(auc_matrix)
    std_array = np.array(std_matrix)
    
    # Create heatmap with AUC values
    cmap = plt.cm.RdYlGn
    im = ax.imshow(auc_array, cmap=cmap, aspect='auto', vmin=0.5, vmax=0.85)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(len(separator_labels_short)))
    ax.set_xticklabels(separator_labels_short, fontsize=11, rotation=0)
    ax.set_yticks(range(len(author_labels)))
    
    # Create y-axis labels with grey text for Human and GPT
    y_labels_objects = []
    for label in author_labels:
        if label in ['Human', 'GPT']:
            y_labels_objects.append(f"{label} †")
        else:
            y_labels_objects.append(label)
    ax.set_yticklabels(y_labels_objects, fontsize=11)
    
    # Make Human and GPT labels grey
    for i, (tick, label) in enumerate(zip(ax.get_yticklabels(), author_labels)):
        if label in ['Human', 'GPT']:
            tick.set_color('#888888')  # Grey color
    
    # Add text annotations with AUC and SE
    for i in range(len(author_labels)):
        row_values = auc_matrix[i]
        sorted_indices = np.argsort(row_values)[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            auc_val = auc_array[i, j]
            std_val = std_array[i, j]
            
            # Determine text weight (bold for best only)
            weight = 'bold' if j == best_idx else 'normal'
            
            # Format text with larger font
            text = f"{auc_val:.3f}\n±{std_val:.3f}"
            
            # All text in black (don't grey out cell values)
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=10, fontweight=weight, color='black')
    
    # Axis labels
    ax.set_xlabel('Perplexity Evaluator Model (P)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Text Generator (Author)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Title
    ax.set_title(f'Author-Specific Separation Performance\n{dataset} Dataset', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(separator_labels_short), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(author_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add separator line between models-as-P and others (after R1, before Human)
    ax.axhline(y=2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add footnote
    footnote = "Bold values indicate best separator for each author. † Not used as separator."
    plt.figtext(0.5, 0.001, footnote, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_author_separation_heatmap.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Create DataFrames
    results_df = pd.DataFrame(results_table, columns=columns)
    numeric_df = pd.DataFrame(auc_matrix, columns=separator_labels_short, 
                             index=[author_labels_full[i] for i in author_order])
    
    return results_table, results_df, numeric_df

def compare_author_vs_others_verbose(paths, n_bootstrap=100, save_prefix=None):
    """
    Verbose appendix version with all metrics (AUC, JS, KL) for Experiment 2: Author-Specific Separation.
    Shows comprehensive performance metrics across different perplexity evaluators.
    Best values are shown in BOLD for all metrics.
    
    :param paths: Nested list of response paths organized by LM separators
    :param n_bootstrap: Number of bootstrap iterations for standard error calculation
    :param save_prefix: Optional prefix for saving heatmap images
    :return: Results tables and DataFrames for all three metrics
    """
    # Reorder authors: models used as both P and generators first
    author_order = [0, 1, 4, 2, 3]  # Llama3.1, Falcon, R1, Human, GPT
    author_labels_full = ['Llama3.1', 'Falcon', 'Human', 'GPT', 'R1']
    author_labels = [author_labels_full[i] for i in author_order]
    
    separator_labels_short = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_labels_full = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Collect numeric values for heatmaps
    auc_matrix = []
    std_matrix = []
    js_matrix = []
    kl_matrix = []
    
    for author_idx in author_order:
        author_label = author_labels_full[author_idx]
        
        row_aucs = []
        row_stds = []
        row_js = []
        row_kl = []
        
        for separator_group in paths:
            aucs, stds, js_dists, kl_divs = [], [], [], []
            
            # Compare this author against all others
            for other_idx in range(len(author_labels_full)):
                if other_idx != author_idx:
                    author_path = separator_group[author_idx]
                    other_author_path = separator_group[other_idx]
                    
                    # AUC with standard error
                    auth1_df = pd.read_csv(author_path)
                    auth2_df = pd.read_csv(other_author_path)
                    mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
                    aucs.append(mean_auc)
                    stds.append(std_error)
                    
                    # JS distance
                    js_dist = calc_js_kde(author_path, other_author_path)
                    js_dists.append(js_dist)
                    
                    # KL divergence (optimal error exponent)
                    kl_div = optimal_error_exponent(author_path, other_author_path)
                    kl_divs.append(kl_div)
            
            # Calculate averages
            avg_auc = np.mean(aucs)
            avg_std = np.sqrt(np.mean(np.array(stds)**2))
            avg_js = np.mean(js_dists)
            avg_kl = np.mean(kl_divs)
            
            row_aucs.append(avg_auc)
            row_stds.append(avg_std)
            row_js.append(avg_js)
            row_kl.append(avg_kl)
        
        auc_matrix.append(row_aucs)
        std_matrix.append(row_stds)
        js_matrix.append(row_js)
        kl_matrix.append(row_kl)
    
    # Create formatted tables (clean, without markers - visual emphasis in heatmap)
    auc_results = []
    js_results = []
    kl_results = []
    
    for i, author_idx in enumerate(author_order):
        author_label = author_labels_full[author_idx]
        
        # Add † for Human/GPT
        if author_label in ['Human', 'GPT']:
            label_with_marker = f"{author_label} †"
        else:
            label_with_marker = author_label
        
        # AUC table
        auc_row = [label_with_marker]
        for j, (auc_val, std_val) in enumerate(zip(auc_matrix[i], std_matrix[i])):
            formatted = f"{auc_val:.3f} ({std_val:.3f})"
            auc_row.append(formatted)
        auc_results.append(auc_row)
        
        # JS table
        js_row = [label_with_marker]
        for j, js_val in enumerate(js_matrix[i]):
            formatted = f"{js_val:.4f}"
            js_row.append(formatted)
        js_results.append(js_row)
        
        # KL table
        kl_row = [label_with_marker]
        for j, kl_val in enumerate(kl_matrix[i]):
            formatted = f"{kl_val:.4f}"
            kl_row.append(formatted)
        kl_results.append(kl_row)
    
    # Print all three tables (LaTeX-style with horizontal lines only)
    columns = ["Text Generator"] + separator_labels_short
    
    print(f"\n=== Experiment 2: Author-Specific Separation - Appendix ({dataset} Dataset) ===")
    print(f"Y-axis: Text generator  |  X-axis: Perplexity evaluator (P)")
    print(f"Best values highlighted in heatmaps  |  † = Not used as separator\n")
    
    print("Table 1: AUC with Standard Errors")
    print(tabulate(auc_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    print(f"\nTable 2: Jensen-Shannon Distance")
    print(tabulate(js_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    print(f"\nTable 3: KL Divergence (Optimal Error Exponent)")
    print(tabulate(kl_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    # Create comprehensive heatmap with all three metrics (slimmer rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, len(author_labels)*0.6 + 1))
    
    auc_array = np.array(auc_matrix)
    js_array = np.array(js_matrix)
    kl_array = np.array(kl_matrix)
    
    # Create y-axis labels with markers
    y_labels_display = []
    for label in author_labels:
        if label in ['Human', 'GPT']:
            y_labels_display.append(f"{label} †")
        else:
            y_labels_display.append(label)
    
    # 1. AUC Heatmap
    im1 = axes[0].imshow(auc_array, cmap=plt.cm.RdYlGn, aspect='auto', vmin=0.5, vmax=0.85)
    axes[0].set_title('AUC', fontsize=12, fontweight='bold')
    
    # Add values with bold for best only
    for i in range(len(author_labels)):
        sorted_indices = np.argsort(auc_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[0].text(j, i, f"{auc_array[i, j]:.3f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[0].set_xticks(range(len(separator_labels_short)))
    axes[0].set_xticklabels(separator_labels_short, rotation=15)
    axes[0].set_yticks(range(len(author_labels)))
    axes[0].set_yticklabels(y_labels_display)
    axes[0].set_ylabel('Text Generator', fontsize=11, fontweight='bold')
    
    # Grey out Human and GPT labels
    for i, label in enumerate(author_labels):
        if label in ['Human', 'GPT']:
            axes[0].get_yticklabels()[i].set_color('#888888')
    
    # Add separator line
    axes[0].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 2. JS Distance Heatmap
    im2 = axes[1].imshow(js_array, cmap=plt.cm.Reds, aspect='auto')
    axes[1].set_title('JS Distance', fontsize=12, fontweight='bold')
    
    for i in range(len(author_labels)):
        sorted_indices = np.argsort(js_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[1].text(j, i, f"{js_array[i, j]:.4f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[1].set_xticks(range(len(separator_labels_short)))
    axes[1].set_xticklabels(separator_labels_short, rotation=15)
    axes[1].set_yticks(range(len(author_labels)))
    axes[1].set_yticklabels([])  # Hide y-labels for middle plot
    axes[1].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 3. KL Divergence Heatmap
    im3 = axes[2].imshow(kl_array, cmap=plt.cm.Purples, aspect='auto')
    axes[2].set_title('KL Divergence', fontsize=12, fontweight='bold')
    
    for i in range(len(author_labels)):
        sorted_indices = np.argsort(kl_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[2].text(j, i, f"{kl_array[i, j]:.4f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[2].set_xticks(range(len(separator_labels_short)))
    axes[2].set_xticklabels(separator_labels_short, rotation=15)
    axes[2].set_yticks(range(len(author_labels)))
    axes[2].set_yticklabels([])  # Hide y-labels for right plot
    axes[2].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Common x-label for all subplots
    fig.text(0.5, 0.04, 'Perplexity Evaluator Model (P)', ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Experiment 2: Author-Specific Separation - All Metrics\n{dataset} Dataset (Appendix)', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Add footnote
    footnote = "Bold values indicate best separator for each author. † Not used as separator."
    plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_author_separation_all_metrics.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Create DataFrames
    auc_df = pd.DataFrame(auc_results, columns=columns)
    js_df = pd.DataFrame(js_results, columns=columns)
    kl_df = pd.DataFrame(kl_results, columns=columns)
    
    return auc_results, js_results, kl_results, auc_df, js_df, kl_df


def compare_author_vs_others_experiment2_clean(paths, n_bootstrap=100, save_prefix=None):
    """
    [ALTERNATIVE VERSION]
    Clean publication-ready table with heatmap for Experiment 2: Author-Specific Separation Performance.
    Shows how well each text generator can be distinguished from others using different perplexity evaluators (P).
    Best values are shown in BOLD, second-best are underlined.
    
    :param paths: Nested list of response paths organized by LM separators
    :param n_bootstrap: Number of bootstrap iterations for standard error calculation
    :param save_prefix: Optional prefix for saving heatmap image
    :return: Results table, DataFrame, and numeric values DataFrame
    """
    # Reorder authors: models used as both P and generators first (Llama3.1, Falcon, R1), then others (Human, GPT)
    author_order = [0, 1, 4, 2, 3]  # Llama3.1, Falcon, R1, Human, GPT
    author_labels_full = ['Llama3.1', 'Falcon', 'Human', 'GPT', 'R1']
    author_labels = [author_labels_full[i] for i in author_order]
    
    # Short labels for table columns
    separator_labels_short = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_labels_full = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    columns = ["Text Generator"] + separator_labels_short
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Store numeric values for heatmap and ranking
    auc_matrix = []
    std_matrix = []
    results_table = []
    
    for author_idx in author_order:
        author_label = author_labels_full[author_idx]
        row_aucs = []
        row_stds = []
        
        for separator_group in paths:
            aucs = []
            stds = []
            
            # Compare this author against all others
            for other_idx in range(len(author_labels_full)):
                if other_idx != author_idx:
                    author_path = separator_group[author_idx]
                    other_author_path = separator_group[other_idx]
                    auth1_df = pd.read_csv(author_path)
                    auth2_df = pd.read_csv(other_author_path)
                    mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
                    aucs.append(mean_auc)
                    stds.append(std_error)
            
            # Calculate average AUC and pooled standard error
            avg_auc = np.mean(aucs)
            avg_std = np.sqrt(np.mean(np.array(stds)**2))
            row_aucs.append(avg_auc)
            row_stds.append(avg_std)
        
        auc_matrix.append(row_aucs)
        std_matrix.append(row_stds)
    
    # Create table (clean, without formatting markers)
    for i, author_idx in enumerate(author_order):
        author_label = author_labels_full[author_idx]
        
        # Grey out Human and GPT (they're not used as separators)
        if author_label in ['Human', 'GPT']:
            row_results = [f"{author_label} †"]
        else:
            row_results = [author_label]
        
        # Just show values without formatting markers (heatmap provides visual emphasis)
        for j, (auc_val, std_val) in enumerate(zip(auc_matrix[i], std_matrix[i])):
            formatted = f"{auc_val:.3f} ({std_val:.3f})"
            row_results.append(formatted)
        
        results_table.append(row_results)
    
    # Print table (LaTeX-style)
    print(f"\n=== Experiment 2: Author-Specific Separation Performance ({dataset} Dataset) ===")
    print(f"Y-axis: Text generator to be distinguished from all others")
    print(f"X-axis: Perplexity evaluator model (P) used for separation")
    print(f"Values: AUC ± Standard Error (best values highlighted in heatmap)")
    print(f"† = Not used as separator (informative only)\n")
    print(tabulate(results_table, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    # Create heatmap with slimmer rows
    fig, ax = plt.subplots(1, 1, figsize=(10, len(author_labels)*0.6 + 1.5))
    
    auc_array = np.array(auc_matrix)
    std_array = np.array(std_matrix)
    
    # Create heatmap with AUC values
    cmap = plt.cm.RdYlGn
    im = ax.imshow(auc_array, cmap=cmap, aspect='auto', vmin=0.5, vmax=0.85)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(len(separator_labels_short)))
    ax.set_xticklabels(separator_labels_short, fontsize=11, rotation=0)
    ax.set_yticks(range(len(author_labels)))
    
    # Create y-axis labels with grey text for Human and GPT
    y_labels_objects = []
    for label in author_labels:
        if label in ['Human', 'GPT']:
            y_labels_objects.append(f"{label} †")
        else:
            y_labels_objects.append(label)
    ax.set_yticklabels(y_labels_objects, fontsize=11)
    
    # Make Human and GPT labels grey
    for i, (tick, label) in enumerate(zip(ax.get_yticklabels(), author_labels)):
        if label in ['Human', 'GPT']:
            tick.set_color('#888888')  # Grey color
    
    # Add text annotations with AUC and SE
    for i in range(len(author_labels)):
        row_values = auc_matrix[i]
        sorted_indices = np.argsort(row_values)[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            auc_val = auc_array[i, j]
            std_val = std_array[i, j]
            
            # Determine text weight (bold for best only)
            weight = 'bold' if j == best_idx else 'normal'
            
            # Format text with larger font
            text = f"{auc_val:.3f}\n±{std_val:.3f}"
            
            # All text in black (don't grey out cell values)
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=10, fontweight=weight, color='black')
    
    # Axis labels
    ax.set_xlabel('Perplexity Evaluator Model (P)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Text Generator (Author)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Title
    ax.set_title(f'Author-Specific Separation Performance\n{dataset} Dataset', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(separator_labels_short), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(author_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add separator line between models-as-P and others (after R1, before Human)
    ax.axhline(y=2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add footnote
    footnote = "Bold values indicate best separator for each author. † Not used as separator."
    plt.figtext(0.5, 0.001, footnote, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_author_separation_heatmap.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Create DataFrames
    results_df = pd.DataFrame(results_table, columns=columns)
    numeric_df = pd.DataFrame(auc_matrix, columns=separator_labels_short, 
                             index=[author_labels_full[i] for i in author_order])
    
    return results_table, results_df, numeric_df


def compare_author_vs_others_experiment2_verbose(paths, n_bootstrap=100, save_prefix=None):
    """
    [AI ASSISTANT METHOD]
    Verbose appendix version with all metrics (AUC, JS, KL) for Experiment 2: Author-Specific Separation.
    Shows comprehensive performance metrics across different perplexity evaluators.
    Best values are shown in BOLD for all metrics.
    
    :param paths: Nested list of response paths organized by LM separators
    :param n_bootstrap: Number of bootstrap iterations for standard error calculation
    :param save_prefix: Optional prefix for saving heatmap images
    :return: Results tables and DataFrames for all three metrics
    """
    # Reorder authors: models used as both P and generators first
    author_order = [0, 1, 4, 2, 3]  # Llama3.1, Falcon, R1, Human, GPT
    author_labels_full = ['Llama3.1', 'Falcon', 'Human', 'GPT', 'R1']
    author_labels = [author_labels_full[i] for i in author_order]
    
    separator_labels_short = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_labels_full = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Collect numeric values for heatmaps
    auc_matrix = []
    std_matrix = []
    js_matrix = []
    kl_matrix = []
    
    for author_idx in author_order:
        author_label = author_labels_full[author_idx]
        
        row_aucs = []
        row_stds = []
        row_js = []
        row_kl = []
        
        for separator_group in paths:
            aucs, stds, js_dists, kl_divs = [], [], [], []
            
            # Compare this author against all others
            for other_idx in range(len(author_labels_full)):
                if other_idx != author_idx:
                    author_path = separator_group[author_idx]
                    other_author_path = separator_group[other_idx]
                    
                    # AUC with standard error
                    auth1_df = pd.read_csv(author_path)
                    auth2_df = pd.read_csv(other_author_path)
                    mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
                    aucs.append(mean_auc)
                    stds.append(std_error)
                    
                    # JS distance
                    js_dist = calc_js_kde(author_path, other_author_path)
                    js_dists.append(js_dist)
                    
                    # KL divergence (optimal error exponent)
                    kl_div = optimal_error_exponent(author_path, other_author_path)
                    kl_divs.append(kl_div)
            
            # Calculate averages
            avg_auc = np.mean(aucs)
            avg_std = np.sqrt(np.mean(np.array(stds)**2))
            avg_js = np.mean(js_dists)
            avg_kl = np.mean(kl_divs)
            
            row_aucs.append(avg_auc)
            row_stds.append(avg_std)
            row_js.append(avg_js)
            row_kl.append(avg_kl)
        
        auc_matrix.append(row_aucs)
        std_matrix.append(row_stds)
        js_matrix.append(row_js)
        kl_matrix.append(row_kl)
    
    # Create formatted tables (clean, without markers - visual emphasis in heatmap)
    auc_results = []
    js_results = []
    kl_results = []
    
    for i, author_idx in enumerate(author_order):
        author_label = author_labels_full[author_idx]
        
        # Add † for Human/GPT
        if author_label in ['Human', 'GPT']:
            label_with_marker = f"{author_label} †"
        else:
            label_with_marker = author_label
        
        # AUC table
        auc_row = [label_with_marker]
        for j, (auc_val, std_val) in enumerate(zip(auc_matrix[i], std_matrix[i])):
            formatted = f"{auc_val:.3f} ({std_val:.3f})"
            auc_row.append(formatted)
        auc_results.append(auc_row)
        
        # JS table
        js_row = [label_with_marker]
        for j, js_val in enumerate(js_matrix[i]):
            formatted = f"{js_val:.4f}"
            js_row.append(formatted)
        js_results.append(js_row)
        
        # KL table
        kl_row = [label_with_marker]
        for j, kl_val in enumerate(kl_matrix[i]):
            formatted = f"{kl_val:.4f}"
            kl_row.append(formatted)
        kl_results.append(kl_row)
    
    # Print all three tables (LaTeX-style)
    columns = ["Text Generator"] + separator_labels_short
    
    print(f"\n=== Experiment 2: Author-Specific Separation - Appendix ({dataset} Dataset) ===")
    print(f"Y-axis: Text generator  |  X-axis: Perplexity evaluator (P)")
    print(f"Best values highlighted in heatmaps  |  † = Not used as separator\n")
    
    print("Table 1: AUC with Standard Errors")
    print(tabulate(auc_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    print(f"\nTable 2: Jensen-Shannon Distance")
    print(tabulate(js_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    print(f"\nTable 3: KL Divergence (Optimal Error Exponent)")
    print(tabulate(kl_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    # Create comprehensive heatmap with all three metrics (slimmer rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, len(author_labels)*0.6 + 1))
    
    auc_array = np.array(auc_matrix)
    js_array = np.array(js_matrix)
    kl_array = np.array(kl_matrix)
    
    # Create y-axis labels with markers
    y_labels_display = []
    for label in author_labels:
        if label in ['Human', 'GPT']:
            y_labels_display.append(f"{label} †")
        else:
            y_labels_display.append(label)
    
    # 1. AUC Heatmap
    im1 = axes[0].imshow(auc_array, cmap=plt.cm.RdYlGn, aspect='auto', vmin=0.5, vmax=0.85)
    axes[0].set_title('AUC', fontsize=12, fontweight='bold')
    
    # Add values with bold for best only
    for i in range(len(author_labels)):
        sorted_indices = np.argsort(auc_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[0].text(j, i, f"{auc_array[i, j]:.3f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[0].set_xticks(range(len(separator_labels_short)))
    axes[0].set_xticklabels(separator_labels_short, rotation=15)
    axes[0].set_yticks(range(len(author_labels)))
    axes[0].set_yticklabels(y_labels_display)
    axes[0].set_ylabel('Text Generator', fontsize=11, fontweight='bold')
    
    # Grey out Human and GPT labels
    for i, label in enumerate(author_labels):
        if label in ['Human', 'GPT']:
            axes[0].get_yticklabels()[i].set_color('#888888')
    
    # Add separator line
    axes[0].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 2. JS Distance Heatmap
    im2 = axes[1].imshow(js_array, cmap=plt.cm.Reds, aspect='auto')
    axes[1].set_title('JS Distance', fontsize=12, fontweight='bold')
    
    for i in range(len(author_labels)):
        sorted_indices = np.argsort(js_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[1].text(j, i, f"{js_array[i, j]:.4f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[1].set_xticks(range(len(separator_labels_short)))
    axes[1].set_xticklabels(separator_labels_short, rotation=15)
    axes[1].set_yticks(range(len(author_labels)))
    axes[1].set_yticklabels([])  # Hide y-labels for middle plot
    axes[1].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 3. KL Divergence Heatmap
    im3 = axes[2].imshow(kl_array, cmap=plt.cm.Purples, aspect='auto')
    axes[2].set_title('KL Divergence', fontsize=12, fontweight='bold')
    
    for i in range(len(author_labels)):
        sorted_indices = np.argsort(kl_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[2].text(j, i, f"{kl_array[i, j]:.4f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[2].set_xticks(range(len(separator_labels_short)))
    axes[2].set_xticklabels(separator_labels_short, rotation=15)
    axes[2].set_yticks(range(len(author_labels)))
    axes[2].set_yticklabels([])  # Hide y-labels for right plot
    axes[2].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Common x-label for all subplots
    fig.text(0.5, 0.04, 'Perplexity Evaluator Model (P)', ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Experiment 2: Author-Specific Separation - All Metrics\n{dataset} Dataset (Appendix)', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Add footnote
    footnote = "Bold values indicate best separator for each author. † Not used as separator."
    plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_author_separation_all_metrics.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Create DataFrames
    auc_df = pd.DataFrame(auc_results, columns=columns)
    js_df = pd.DataFrame(js_results, columns=columns)
    kl_df = pd.DataFrame(kl_results, columns=columns)
    
    return auc_results, js_results, kl_results, auc_df, js_df, kl_df


# ==============================================================================
# EXPERIMENT 3: Human vs AI-Generated Text Detection
# Clean and verbose versions with professional formatting and heatmaps
# ==============================================================================

def human_vs_ai_detection_experiment3_clean(paths, n_bootstrap=100, save_prefix=None):
    """
    [AI ASSISTANT METHOD]
    Clean publication-ready table with heatmap for Experiment 3: Human vs AI-Generated Text Detection.
    Shows how well each LM separator can distinguish human text from AI-generated text.
    Best values are shown in BOLD.
    
    :param paths: Nested list of response paths organized by LM separators
    :param n_bootstrap: Number of bootstrap iterations for standard error calculation
    :param save_prefix: Optional prefix for saving heatmap image
    :return: Results table, DataFrame, and numeric values DataFrame
    """
    # Reorder: models used as separators first (Llama3.1, Falcon, R1), then GPT at the end (not used as separator)
    comparisons = [
        ("Human", "Llama3.1", 0),
        ("Human", "Falcon", 1),
        ("Human", "R1", 4),
        ("Human", "GPT", 3)  # GPT at the end - not used as separator
    ]
    
    separator_labels_short = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_labels_full = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    columns = ["Comparison"] + separator_labels_short
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Collect numeric values for heatmap
    auc_matrix = []
    std_matrix = []
    results_table = []
    comparison_labels = []
    
    for human_label, gen_label, idx in comparisons:
        row_aucs = []
        row_stds = []
        
        for lm_response_group in paths:
            auth1_path = lm_response_group[2]   # human responses
            auth2_path = lm_response_group[idx]  # LLM responses
            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
            row_aucs.append(mean_auc)
            row_stds.append(std_error)
        
        auc_matrix.append(row_aucs)
        std_matrix.append(row_stds)
        comparison_labels.append(f"{human_label} vs {gen_label}")
    
    # Create table (clean, without formatting markers - add † for GPT)
    for i, (label, (_, gen_label, _)) in enumerate(zip(comparison_labels, comparisons)):
        # Grey out GPT (it's not used as a separator)
        if gen_label == "GPT":
            row_results = [f"{label} †"]
        else:
            row_results = [label]
        
        for j, (auc_val, std_val) in enumerate(zip(auc_matrix[i], std_matrix[i])):
            formatted = f"{auc_val:.3f} ({std_val:.3f})"
            row_results.append(formatted)
        results_table.append(row_results)
    
    # Print table (LaTeX-style)
    print(f"\n=== Experiment 3: Human vs AI-Generated Text Detection ({dataset} Dataset) ===")
    print(f"Rows: Human text vs specific AI model")
    print(f"Columns: Perplexity evaluator model (P) used for detection")
    print(f"Values: AUC ± Standard Error (best values highlighted in heatmap)")
    print(f"† = Not used as separator (informative only)\n")
    print(tabulate(results_table, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, len(comparison_labels)*0.6 + 1.5))
    
    auc_array = np.array(auc_matrix)
    std_array = np.array(std_matrix)
    
    # Create heatmap with AUC values
    cmap = plt.cm.RdYlGn
    im = ax.imshow(auc_array, cmap=cmap, aspect='auto', vmin=0.5, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUC', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(len(separator_labels_short)))
    ax.set_xticklabels(separator_labels_short, fontsize=11, rotation=0)
    ax.set_yticks(range(len(comparison_labels)))
    
    # Create y-axis labels with † marker for GPT
    y_labels_display = []
    for i, (_, gen_label, _) in enumerate(comparisons):
        label = comparison_labels[i]
        if gen_label == "GPT":
            y_labels_display.append(f"{label} †")
        else:
            y_labels_display.append(label)
    ax.set_yticklabels(y_labels_display, fontsize=11)
    
    # Make GPT label grey
    for i, (_, gen_label, _) in enumerate(comparisons):
        if gen_label == "GPT":
            ax.get_yticklabels()[i].set_color('#888888')  # Grey color
    
    # Add text annotations with AUC and SE
    for i in range(len(comparison_labels)):
        row_values = auc_matrix[i]
        sorted_indices = np.argsort(row_values)[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            auc_val = auc_array[i, j]
            std_val = std_array[i, j]
            
            # Bold for best
            weight = 'bold' if j == best_idx else 'normal'
            
            # Format text
            text = f"{auc_val:.3f}\n±{std_val:.3f}"
            
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=10, fontweight=weight, color='black')
    
    # Axis labels
    ax.set_xlabel('Perplexity Evaluator Model (P)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Human vs AI Comparison', fontsize=12, fontweight='bold', labelpad=10)
    
    # Title
    ax.set_title(f'Human vs AI-Generated Text Detection\n{dataset} Dataset', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(separator_labels_short), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(comparison_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add separator line between models-as-separators and GPT (after R1, before GPT)
    ax.axhline(y=2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add footnote
    footnote = "Bold values indicate best detector for each comparison. † Not used as separator."
    plt.figtext(0.5, 0.001, footnote, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_human_vs_ai_heatmap.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Create DataFrames
    results_df = pd.DataFrame(results_table, columns=columns)
    numeric_df = pd.DataFrame(auc_matrix, columns=separator_labels_short, 
                             index=comparison_labels)
    
    return results_table, results_df, numeric_df


def human_vs_ai_detection_experiment3_verbose(paths, n_bootstrap=100, save_prefix=None):
    """
    [AI ASSISTANT METHOD]
    Verbose appendix version with all metrics (AUC, JS, KL) for Experiment 3: Human vs AI Detection.
    Shows comprehensive performance metrics across different perplexity evaluators.
    Best values are shown in BOLD for all metrics.
    
    :param paths: Nested list of response paths organized by LM separators
    :param n_bootstrap: Number of bootstrap iterations for standard error calculation
    :param save_prefix: Optional prefix for saving heatmap images
    :return: Results tables and DataFrames for all three metrics
    """
    # Reorder: models used as separators first (Llama3.1, Falcon, R1), then GPT at the end (not used as separator)
    comparisons = [
        ("Human", "Llama3.1", 0),
        ("Human", "Falcon", 1),
        ("Human", "R1", 4),
        ("Human", "GPT", 3)  # GPT at the end - not used as separator
    ]
    
    separator_labels_short = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_labels_full = ["Llama-3.1-8B-Instruct", "Falcon-7b", "Phi-2", "DeepSeek-R1-Distill-Qwen-7B"]
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # Collect numeric values for heatmaps
    auc_matrix = []
    std_matrix = []
    js_matrix = []
    kl_matrix = []
    comparison_labels = []
    
    for human_label, gen_label, idx in comparisons:
        row_aucs = []
        row_stds = []
        row_js = []
        row_kl = []
        
        for lm_response_group in paths:
            auth1_path = lm_response_group[2]   # human responses
            auth2_path = lm_response_group[idx]  # LLM responses
            
            # AUC with standard error
            auth1_df = pd.read_csv(auth1_path)
            auth2_df = pd.read_csv(auth2_path)
            mean_auc, std_error = bootstrap_auc_clean(auth1_df, auth2_df, n_bootstrap)
            row_aucs.append(mean_auc)
            row_stds.append(std_error)
            
            # JS distance
            js_dist = calc_js_kde(auth1_path, auth2_path)
            row_js.append(js_dist)
            
            # KL divergence
            kl_div = optimal_error_exponent(auth1_path, auth2_path)
            row_kl.append(kl_div)
        
        auc_matrix.append(row_aucs)
        std_matrix.append(row_stds)
        js_matrix.append(row_js)
        kl_matrix.append(row_kl)
        comparison_labels.append(f"{human_label} vs {gen_label}")
    
    # Create formatted tables (add † for GPT)
    auc_results = []
    js_results = []
    kl_results = []
    
    for i, (label, (_, gen_label, _)) in enumerate(zip(comparison_labels, comparisons)):
        # Add † marker for GPT (not used as separator)
        if gen_label == "GPT":
            display_label = f"{label} †"
        else:
            display_label = label
        
        # AUC table
        auc_row = [display_label]
        for j, (auc_val, std_val) in enumerate(zip(auc_matrix[i], std_matrix[i])):
            formatted = f"{auc_val:.3f} ({std_val:.3f})"
            auc_row.append(formatted)
        auc_results.append(auc_row)
        
        # JS table
        js_row = [display_label]
        for j, js_val in enumerate(js_matrix[i]):
            formatted = f"{js_val:.4f}"
            js_row.append(formatted)
        js_results.append(js_row)
        
        # KL table
        kl_row = [display_label]
        for j, kl_val in enumerate(kl_matrix[i]):
            formatted = f"{kl_val:.4f}"
            kl_row.append(formatted)
        kl_results.append(kl_row)
    
    # Print all three tables (LaTeX-style)
    columns = ["Comparison"] + separator_labels_short
    
    print(f"\n=== Experiment 3: Human vs AI Detection - Appendix ({dataset} Dataset) ===")
    print(f"Rows: Human vs AI model  |  Columns: Perplexity evaluator (P)")
    print(f"Best values highlighted in heatmaps  |  † = Not used as separator\n")
    
    print("Table 1: AUC with Standard Errors")
    print(tabulate(auc_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    print(f"\nTable 2: Jensen-Shannon Distance")
    print(tabulate(js_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    print(f"\nTable 3: KL Divergence (Optimal Error Exponent)")
    print(tabulate(kl_results, headers=columns, tablefmt="simple", colalign=("center",)*(len(columns))))
    
    # Create comprehensive heatmap with all three metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, len(comparison_labels)*0.6 + 1))
    
    auc_array = np.array(auc_matrix)
    js_array = np.array(js_matrix)
    kl_array = np.array(kl_matrix)
    
    # Create y-axis labels with † marker for GPT
    y_labels_display = []
    for i, (_, gen_label, _) in enumerate(comparisons):
        label = comparison_labels[i]
        if gen_label == "GPT":
            y_labels_display.append(f"{label} †")
        else:
            y_labels_display.append(label)
    
    # 1. AUC Heatmap
    im1 = axes[0].imshow(auc_array, cmap=plt.cm.RdYlGn, aspect='auto', vmin=0.5, vmax=1.0)
    axes[0].set_title('AUC', fontsize=12, fontweight='bold')
    
    for i in range(len(comparison_labels)):
        sorted_indices = np.argsort(auc_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[0].text(j, i, f"{auc_array[i, j]:.3f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[0].set_xticks(range(len(separator_labels_short)))
    axes[0].set_xticklabels(separator_labels_short, rotation=15)
    axes[0].set_yticks(range(len(comparison_labels)))
    axes[0].set_yticklabels(y_labels_display)
    axes[0].set_ylabel('Human vs AI', fontsize=11, fontweight='bold')
    
    # Grey out GPT label
    for i, (_, gen_label, _) in enumerate(comparisons):
        if gen_label == "GPT":
            axes[0].get_yticklabels()[i].set_color('#888888')
    
    # Add separator line before GPT (after R1)
    axes[0].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 2. JS Distance Heatmap
    im2 = axes[1].imshow(js_array, cmap=plt.cm.Reds, aspect='auto')
    axes[1].set_title('JS Distance', fontsize=12, fontweight='bold')
    
    for i in range(len(comparison_labels)):
        sorted_indices = np.argsort(js_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[1].text(j, i, f"{js_array[i, j]:.4f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[1].set_xticks(range(len(separator_labels_short)))
    axes[1].set_xticklabels(separator_labels_short, rotation=15)
    axes[1].set_yticks(range(len(comparison_labels)))
    axes[1].set_yticklabels([])  # Hide y-labels for middle plot
    axes[1].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 3. KL Divergence Heatmap
    im3 = axes[2].imshow(kl_array, cmap=plt.cm.Purples, aspect='auto')
    axes[2].set_title('KL Divergence', fontsize=12, fontweight='bold')
    
    for i in range(len(comparison_labels)):
        sorted_indices = np.argsort(kl_matrix[i])[::-1]
        best_idx = sorted_indices[0]
        
        for j in range(len(separator_labels_short)):
            weight = 'bold' if j == best_idx else 'normal'
            axes[2].text(j, i, f"{kl_array[i, j]:.4f}", 
                        ha='center', va='center', fontsize=10, 
                        fontweight=weight, color='black')
    
    axes[2].set_xticks(range(len(separator_labels_short)))
    axes[2].set_xticklabels(separator_labels_short, rotation=15)
    axes[2].set_yticks(range(len(comparison_labels)))
    axes[2].set_yticklabels([])  # Hide y-labels for right plot
    axes[2].axhline(y=2.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Common x-label
    fig.text(0.5, 0.04, 'Perplexity Evaluator Model (P)', ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Experiment 3: Human vs AI Detection - All Metrics\n{dataset} Dataset (Appendix)', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Add footnote
    footnote = "Bold values indicate best detector for each comparison. † Not used as separator."
    plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if save_prefix:
        plt.savefig(f"images/{save_prefix}_human_vs_ai_all_metrics.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    # Create DataFrames
    auc_df = pd.DataFrame(auc_results, columns=columns)
    js_df = pd.DataFrame(js_results, columns=columns)
    kl_df = pd.DataFrame(kl_results, columns=columns)
    
    return auc_results, js_results, kl_results, auc_df, js_df, kl_df


################################################################################
### END OF AI ASSISTANT METHODS                                             ###
################################################################################
