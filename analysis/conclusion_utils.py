"""
Conclusion Analysis Utilities

High-level aggregate analysis across all experiments to demonstrate
the superiority of self-detectors (where LM is one of the authors).

Key metrics:
1. Paired AUC lift (Primary): Median improvement when using self-detector vs. best non-self
2. Win-rate (Supplementary): How often does the best detector come from self vs. non-self?
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from analysis_utils import bootstrap_auc_clean, extract_info_from_path


def aggregate_all_pairs_with_self_flag(paths):
    """
    Aggregate all author pairs across all separators with is_self flag.
    
    For each author pair and each separator:
    - Compute AUC
    - Flag whether separator is one of the two authors (is_self=True)
    
    :param paths: Nested list of response paths organized by LM separators
    :return: DataFrame with columns: 
             [dataset, author1, author2, separator, auc, std_error, is_self]
    """
    # Author labels in order: Llama3.1, Falcon, Human, GPT, R1
    author_labels = ['Llama3.1', 'Falcon', 'Human', 'GPT', 'R1']
    separator_labels = ["Llama-3.1", "Falcon", "Phi-2", "DeepSeek-R1"]
    separator_to_author = {
        "Llama-3.1": "Llama3.1",
        "Falcon": "Falcon",
        "Phi-2": None,  # Not an author
        "DeepSeek-R1": "R1"
    }
    
    results = []
    
    # Extract dataset name
    dataset, _, _, _ = extract_info_from_path(paths[0][0])
    
    # For each separator
    for sep_idx, separator in enumerate(separator_labels):
        separator_author = separator_to_author[separator]
        
        # For each author pair
        for i, author1 in enumerate(author_labels):
            for j, author2 in enumerate(author_labels):
                if i < j:  # Only consider each pair once
                    # Read the data
                    path1 = paths[sep_idx][i]
                    path2 = paths[sep_idx][j]
                    df1 = pd.read_csv(path1)
                    df2 = pd.read_csv(path2)
                    
                    # Compute AUC with bootstrap
                    auc, std_error = bootstrap_auc_clean(df1, df2, n_bootstrap=200)
                    
                    # Check if separator is one of the authors (self-detector)
                    is_self = separator_author in [author1, author2]
                    
                    results.append({
                        'dataset': dataset,
                        'author1': author1,
                        'author2': author2,
                        'pair': f"{author1}-{author2}",
                        'separator': separator,
                        'separator_author': separator_author,
                        'auc': auc,
                        'std_error': std_error,
                        'is_self': is_self
                    })
    
    return pd.DataFrame(results)


def compute_top_auc_winrate(df):
    """
    Compute how often the best AUC for a pair comes from a self-detector.
    
    Only counts pairs where at least one self-detector exists (excludes pairs
    like Human-GPT where neither author is a detector).
    
    :param df: DataFrame from aggregate_all_pairs_with_self_flag
    :return: Dictionary with win-rate statistics and binomial test results
    """
    # For each pair, find the separator with the highest AUC
    pair_best = df.groupby('pair').apply(
        lambda x: x.loc[x['auc'].idxmax()]
    ).reset_index(drop=True)
    
    # Only count pairs that have at least one self-detector possibility
    # (excludes pairs like Human-GPT where neither is a detector)
    analyzable_pairs = []
    for pair_name, group in df.groupby('pair'):
        if group['is_self'].any():  # At least one self-detector exists for this pair
            analyzable_pairs.append(pair_name)
    
    # Filter to only analyzable pairs
    pair_best_analyzable = pair_best[pair_best['pair'].isin(analyzable_pairs)]
    
    # Count wins among analyzable pairs
    n_total = len(pair_best_analyzable)
    n_self_wins = pair_best_analyzable['is_self'].sum()
    n_nonself_wins = n_total - n_self_wins
    
    win_rate = n_self_wins / n_total if n_total > 0 else 0
    
    # Binomial test: H0: p=0.5 (self and non-self equally likely to win)
    binom_result = binomtest(n_self_wins, n_total, p=0.5, alternative='greater')
    
    return {
        'n_pairs': n_total,
        'n_self_wins': n_self_wins,
        'n_nonself_wins': n_nonself_wins,
        'win_rate': win_rate,
        'binom_pvalue': binom_result.pvalue,
        'binom_statistic': binom_result.statistic
    }


def compute_paired_auc_lift(df):
    """
    For each pair, compute: ΔAUC = (mean self AUC) - (mean non-self AUC).
    
    This approach gives equal weight to all separators, reducing the influence of
    any single strong separator (like Phi-2) and aligning with Experiment 1's methodology.
    
    :param df: DataFrame from aggregate_all_pairs_with_self_flag
    :return: Dictionary with lift statistics including Wilcoxon test and significance
    """
    lifts = []
    pair_details = []
    
    for pair_name, group in df.groupby('pair'):
        self_group = group[group['is_self'] == True]
        nonself_group = group[group['is_self'] == False]
        
        if len(self_group) > 0 and len(nonself_group) > 0:
            # Use MEAN instead of MAX to give equal weight to all separators
            mean_self_auc = self_group['auc'].mean()
            mean_nonself_auc = nonself_group['auc'].mean()
            
            delta_auc = mean_self_auc - mean_nonself_auc
            lifts.append(delta_auc)
            
            pair_details.append({
                'pair': pair_name,
                'mean_self_auc': mean_self_auc,
                'mean_nonself_auc': mean_nonself_auc,
                'n_self': len(self_group),
                'n_nonself': len(nonself_group),
                'delta_auc': delta_auc,
                'positive_lift': delta_auc > 0
            })
    
    lifts = np.array(lifts)
    
    # Bootstrap 95% CI for median lift
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    bootstrap_medians = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(lifts, size=len(lifts), replace=True)
        bootstrap_medians.append(np.median(sample))
    
    ci_lower = np.percentile(bootstrap_medians, 2.5)
    ci_upper = np.percentile(bootstrap_medians, 97.5)
    
    # Percentage with positive lift
    pct_positive = np.mean(lifts > 0) * 100
    
    # Wilcoxon signed-rank test: H0: median ΔAUC = 0
    # Tests whether the distribution of lifts is symmetric around zero
    try:
        wilcoxon_result = wilcoxon(lifts, alternative='greater')
        wilcoxon_pvalue = wilcoxon_result.pvalue
    except ValueError:
        # If all values are zero or test cannot be performed
        wilcoxon_pvalue = 1.0
    
    # Statistical significance: CI excludes zero
    is_significant = ci_lower > 0
    
    return {
        'median_lift': np.median(lifts),
        'mean_lift': np.mean(lifts),
        'std_lift': np.std(lifts),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pct_positive': pct_positive,
        'wilcoxon_pvalue': wilcoxon_pvalue,
        'is_significant': is_significant,
        'n_pairs': len(lifts),
        'lifts': lifts,
        'pair_details': pd.DataFrame(pair_details)
    }


def analyze_dataset_conclusion(paths, dataset_name=None):
    """
    Complete conclusion analysis for a single dataset.
    
    :param paths: Nested list of response paths
    :param dataset_name: Optional name override
    :return: Dictionary with all analysis results
    """
    print(f"\n{'='*80}")
    print(f" CONCLUSION ANALYSIS: {dataset_name or 'Dataset'} ".center(80, '='))
    print(f"{'='*80}\n")
    
    # Aggregate all pairs
    print("Aggregating all author pairs with self-detector flags...")
    df = aggregate_all_pairs_with_self_flag(paths)
    
    if dataset_name:
        df['dataset'] = dataset_name
    
    print(f"Total comparisons: {len(df)}")
    print(f"Self-detector comparisons: {df['is_self'].sum()}")
    print(f"Non-self detector comparisons: {(~df['is_self']).sum()}\n")
    
    # Analysis 1 (Primary): Paired AUC lift
    print("-" * 80)
    print("1. PAIRED AUC LIFT ANALYSIS (Primary Metric)")
    print("-" * 80)
    
    lift_results = compute_paired_auc_lift(df)
    
    print(f"\nΔAUC = (Mean Self-Detector AUC) - (Mean Non-Self Detector AUC)")
    print(f"  [Equal weight to all separators, reduces Phi-2 dominance]")
    print(f"  Median ΔAUC: {lift_results['median_lift']:.4f}")
    print(f"  Mean ΔAUC:   {lift_results['mean_lift']:.4f} ± {lift_results['std_lift']:.4f}")
    print(f"  95% CI:      [{lift_results['ci_lower']:.4f}, {lift_results['ci_upper']:.4f}]")
    print(f"  Positive lift: {lift_results['pct_positive']:.1f}% of pairs")
    print(f"  Wilcoxon test (H0: median ΔAUC = 0): p = {lift_results['wilcoxon_pvalue']:.6f}")
    
    # Statistical significance
    if lift_results['is_significant']:
        print(f"\n  STATISTICALLY SIGNIFICANT (95% CI excludes zero)")
        if lift_results['wilcoxon_pvalue'] < 0.001:
            print(f"  Wilcoxon test: Highly significant (p < 0.001)")
        elif lift_results['wilcoxon_pvalue'] < 0.01:
            print(f"  Wilcoxon test: Significant (p < 0.01)")
        elif lift_results['wilcoxon_pvalue'] < 0.05:
            print(f"  Wilcoxon test: Significant (p < 0.05)")
    else:
        print(f"\n  NOT STATISTICALLY SIGNIFICANT (95% CI includes zero)")
    
    # Analysis 2 (Supplementary): Win-rate
    print("\n" + "-" * 80)
    print("2. WIN-RATE ANALYSIS (Supplementary)")
    print("-" * 80)
    
    winrate_results = compute_top_auc_winrate(df)
    
    print(f"\nOut of {winrate_results['n_pairs']} analyzable author pairs:")
    print(f"  (Excludes Human-GPT pair where neither is a detector)")
    print(f"  Self-detector had best AUC:     {winrate_results['n_self_wins']} times ({winrate_results['win_rate']*100:.1f}%)")
    print(f"  Non-self detector had best AUC: {winrate_results['n_nonself_wins']} times ({(1-winrate_results['win_rate'])*100:.1f}%)")
    print(f"  Binomial test (H0: p=0.5): p = {winrate_results['binom_pvalue']:.6f}")
    
    if winrate_results['binom_pvalue'] < 0.001:
        print("    *** Highly significant (p < 0.001)")
    elif winrate_results['binom_pvalue'] < 0.01:
        print("    ** Significant (p < 0.01)")
    elif winrate_results['binom_pvalue'] < 0.05:
        print("    * Significant (p < 0.05)")
    else:
        print("    ns (not significant)")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'df': df,
        'lift': lift_results,
        'winrate': winrate_results
    }


def create_conclusion_summary_table(all_results, save_path=None):
    """
    Create a summary table across all datasets.
    
    :param all_results: Dictionary mapping dataset names to analysis results
    :param save_path: Optional path to save table as image
    :return: DataFrame with summary statistics
    """
    summary_data = []
    
    for dataset_name, results in all_results.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Median ΔAUC': f"{results['lift']['median_lift']:.4f}",
            '95% CI': f"[{results['lift']['ci_lower']:.4f}, {results['lift']['ci_upper']:.4f}]",
            'p-value': f"{results['lift']['wilcoxon_pvalue']:.4f}",
            'Positive Lift': f"{results['lift']['pct_positive']:.1f}%",
            'Win Rate': f"{results['winrate']['win_rate']*100:.1f}%",
            'Pairs': f"{results['winrate']['n_self_wins']}/{results['winrate']['n_pairs']}"
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save as image if path provided
    if save_path:
        fig, ax = plt.subplots(figsize=(14, 2.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with better formatting similar to tabulate
        table = ax.table(cellText=df.values, 
                        colLabels=df.columns,
                        cellLoc='left',  # Left-align for better readability
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Set column widths to prevent text cutoff
        # Columns: Dataset, Median ΔAUC, 95% CI, p-value, Positive Lift, Win Rate, Pairs
        col_widths = [0.12, 0.13, 0.24, 0.11, 0.14, 0.13, 0.13]
        for i, width in enumerate(col_widths):
            for j in range(len(df) + 1):  # +1 for header
                table[(j, i)].set_width(width)
        
        # Header styling
        for i in range(len(df.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(weight='bold', fontsize=9)
            cell.set_height(0.08)
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
        
        # Data row styling - minimal like tabulate
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                cell = table[(i, j)]
                cell.set_facecolor('white')
                cell.set_height(0.08)
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"Table saved to: {save_path}")
    
    return df


def create_aggregate_auc_table(all_dataset_dfs, save_path=None):
    """
    Create a summary table showing mean AUC for self vs non-self detectors,
    aggregated across all pairs in each dataset.
    
    Only includes detectors that are also authors (excludes Phi-2).
    
    :param all_dataset_dfs: Dictionary mapping dataset names to aggregated DataFrames
                            (from aggregate_all_pairs_with_self_flag)
    :param save_path: Optional path to save table as image
    :return: DataFrame with summary statistics
    """
    from scipy import stats
    
    summary_data = []
    
    for dataset_name, df in all_dataset_dfs.items():
        # Exclude Phi-2 from the analysis
        df_filtered = df[df['separator'] != 'Phi-2'].copy()
        
        # Separate self and non-self groups
        self_group = df_filtered[df_filtered['is_self'] == True]
        nonself_group = df_filtered[df_filtered['is_self'] == False]
        
        # Compute statistics for self group
        self_aucs = self_group['auc'].values
        self_mean = self_aucs.mean()
        self_se = stats.sem(self_aucs)  # Standard error
        self_ci_lower = self_mean - 1.96 * self_se  # 95% CI
        self_ci_upper = self_mean + 1.96 * self_se
        self_n = len(self_aucs)
        
        # Compute statistics for non-self group
        nonself_aucs = nonself_group['auc'].values
        nonself_mean = nonself_aucs.mean()
        nonself_se = stats.sem(nonself_aucs)
        nonself_ci_lower = nonself_mean - 1.96 * nonself_se
        nonself_ci_upper = nonself_mean + 1.96 * nonself_se
        nonself_n = len(nonself_aucs)
        
        # Compute delta AUC with its own CI
        delta_auc = self_mean - nonself_mean
        
        # SE of difference for independent samples: sqrt(SE1² + SE2²)
        delta_se = np.sqrt(self_se**2 + nonself_se**2)
        delta_ci_lower = delta_auc - 1.96 * delta_se
        delta_ci_upper = delta_auc + 1.96 * delta_se
        
        summary_data.append({
            'Dataset': dataset_name,
            'Self AUC [95% CI]': f"{self_mean:.3f} [{self_ci_lower:.3f}, {self_ci_upper:.3f}]",
            'Non-Self AUC [95% CI]': f"{nonself_mean:.3f} [{nonself_ci_lower:.3f}, {nonself_ci_upper:.3f}]",
            'ΔAUC [95% CI]': f"{delta_auc:.3f} [{delta_ci_lower:.3f}, {delta_ci_upper:.3f}]",
            'n (Self/Non-Self)': f"{self_n}/{nonself_n}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Print tabulated version
    print("\n" + "="*120)
    print("AGGREGATE AUC ANALYSIS: Self vs Non-Self Detectors (Excluding Phi-2)")
    print("="*120)
    print(tabulate(df_summary, headers='keys', tablefmt='grid', showindex=False))
    print("="*120)
    
    # Generate LaTeX table
    print("\n" + "="*120)
    print("LaTeX Code (for paper):")
    print("="*120)
    
    latex_table = r"""\begin{table}[htbp]
\centering
\caption{Aggregate AUC comparison between self-detectors and non-self detectors across datasets. Phi-2 excluded from analysis. Confidence intervals computed independently for each group and for the difference.}
\label{tab:aggregate_auc}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.2}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}l l l l l@{}}
\hline
"""
    
    # Add header
    latex_table += r"\textbf{Dataset} & \textbf{Self AUC [95\% CI]} & \textbf{Non-Self AUC [95\% CI]} & \textbf{$\Delta$AUC [95\% CI]} & \textbf{n (Self/Non-Self)} \\" + "\n"
    latex_table += r"\hline" + "\n"
    
    # Add data rows
    for _, row in df_summary.iterrows():
        dataset = row['Dataset']
        self_auc = row['Self AUC [95% CI]'].replace('-', '$-$')  # Proper LaTeX minus
        nonself_auc = row['Non-Self AUC [95% CI]'].replace('-', '$-$')
        delta_auc = row['ΔAUC [95% CI]'].replace('-', '$-$')
        n_counts = row['n (Self/Non-Self)']
        
        latex_table += f" {dataset:<9} & {self_auc:<30} & {nonself_auc:<30} & {delta_auc:<30} & {n_counts:<10} \\\\\n"
    
    # Add footer
    latex_table += r"\hline" + "\n"
    latex_table += r"""\end{tabular}%
}
\end{table}"""
    
    print(latex_table)
    print("\n" + "="*120 + "\n")
    
    # Save as image if path provided
    if save_path:
        fig, ax = plt.subplots(figsize=(16, 2.5))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_summary.values,
                        colLabels=df_summary.columns,
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        # Set column widths for new simplified format
        # Columns: Dataset, Self AUC [95% CI], Non-Self AUC [95% CI], ΔAUC [95% CI], n (Self/Non-Self)
        col_widths = [0.12, 0.30, 0.30, 0.22, 0.12]
        for i, width in enumerate(col_widths):
            for j in range(len(df_summary) + 1):
                table[(j, i)].set_width(width)
        
        # Header styling
        for i in range(len(df_summary.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(weight='bold', fontsize=8)
            cell.set_height(0.08)
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
        
        # Data row styling
        for i in range(1, len(df_summary) + 1):
            for j in range(len(df_summary.columns)):
                cell = table[(i, j)]
                cell.set_facecolor('white')
                cell.set_height(0.08)
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        plt.close()
        print(f"Table saved to: {save_path}")
    
    return df_summary


def plot_conclusion_visualizations(all_results, save_prefix=None):
    """
    Create clean, professional visualization with 3 key plots.
    
    :param all_results: Dictionary mapping dataset names to analysis results
    :param save_prefix: Optional prefix for saving plots
    """
    datasets = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    
    # Plot 1: Statistical Significance - Median ΔAUC with 95% CI
    ax1 = axes[0]
    
    medians = [all_results[d]['lift']['median_lift'] for d in datasets]
    ci_lowers = [all_results[d]['lift']['ci_lower'] for d in datasets]
    ci_uppers = [all_results[d]['lift']['ci_upper'] for d in datasets]
    is_sig = [all_results[d]['lift']['is_significant'] for d in datasets]
    
    # Professional colors - blue scheme
    colors = ['#2E75B6' if sig else '#C5C5C5' for sig in is_sig]
    
    x_pos = np.arange(len(datasets))
    bars = ax1.bar(x_pos, medians, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add error bars for 95% CI
    errors_lower = [m - l for m, l in zip(medians, ci_lowers)]
    errors_upper = [u - m for m, u in zip(medians, ci_uppers)]
    ax1.errorbar(x_pos, medians, yerr=[errors_lower, errors_upper], 
                fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, fontsize=11)
    ax1.set_ylabel('Median ΔAUC', fontsize=12, fontweight='bold')
    ax1.set_title('AUC Lift: Mean Self vs Mean Non-Self', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.2, linestyle=':')
    
    # Add p-values as annotations
    for i, (dataset, bar) in enumerate(zip(datasets, bars)):
        p_val = all_results[dataset]['lift']['wilcoxon_pvalue']
        height = bar.get_height()
        y_pos = height + errors_upper[i] + 0.003
        
        if p_val < 0.001:
            p_text = 'p<0.001'
        elif p_val < 0.01:
            p_text = f'p={p_val:.3f}'
        else:
            p_text = f'p={p_val:.2f}'
        
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                p_text, ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Positive Lift Percentage
    ax2 = axes[1]
    
    pct_positive = [all_results[d]['lift']['pct_positive'] for d in datasets]
    colors_pos = ['#2E75B6' if is_sig[i] else '#C5C5C5' for i in range(len(datasets))]
    
    bars = ax2.bar(datasets, pct_positive, color=colors_pos, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (50%)')
    ax2.set_ylabel('% of Pairs', fontsize=12, fontweight='bold')
    ax2.set_title('Pairs with Positive ΔAUC', fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(axis='y', alpha=0.2, linestyle=':')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, pct_positive)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Self-Detector Win Rate
    ax3 = axes[2]
    
    win_rates = [all_results[d]['winrate']['win_rate'] * 100 for d in datasets]
    colors_wr = ['#2E75B6' if is_sig[i] else '#C5C5C5' for i in range(len(datasets))]
    
    bars = ax3.bar(datasets, win_rates, color=colors_wr, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (50%)')
    ax3.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Self-Detector Best AUC Rate', fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylim([0, 105])
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(axis='y', alpha=0.2, linestyle=':')
    
    # Add percentage labels on bars
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{wr:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    plt.suptitle('Self-Detector Performance Analysis', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'images/{save_prefix}_conclusion_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: images/{save_prefix}_conclusion_analysis.png")
    
    plt.show()

