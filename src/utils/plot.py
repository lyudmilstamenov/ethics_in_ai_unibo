import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_mean_std(metric_list):
    """
    Calculate the mean and standard deviation of a list of metric values.

    Parameters:
    metric_list (list): List of numerical metric values.

    Returns:
    tuple: Mean and standard deviation of the input list.
    """
    arr = np.array(metric_list)
    return np.mean(arr), np.std(arr)


def plot_metrics(plot_data, metric_name, repair_levels, protected_attributes):
    """
    Plot mean and standard deviation of a given metric across different repair levels
    for each sensitive attribute using error bars.

    Parameters:
    plot_data (dict): Dictionary containing metric statistics by attribute and repair level.
                      Expected keys format: 
                      plot_data[attribute][f"{metric_name}_mean_{repair_level}"] and 
                      plot_data[attribute][f"{metric_name}_std_{repair_level}"]
    metric_name (str): Name of the metric to plot.
    repair_levels (list): List of repair levels used in the fairness pipeline.
    protected_attributes (list): List of sensitive attributes.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    
    for sensitive_attr in protected_attributes:
        means = [plot_data[sensitive_attr][f"{metric_name}_mean_{rl}"] for rl in repair_levels]
        stds = [plot_data[sensitive_attr][f"{metric_name}_std_{rl}"] for rl in repair_levels]
        
        plt.errorbar(repair_levels, means, yerr=stds, 
                     label=sensitive_attr, marker='o', capsize=5)
    
    plt.xlabel('Repair Level')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} vs Repair Level')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics_grouped(results, protected_attributes, repair_levels):
    """
    Create grouped bar plots of performance and fairness metrics across repair levels
    for each protected attribute. Y-axis limits are synchronized per attribute row
    (performance and fairness), allowing for easy comparison within attributes.
    
    Parameters:
    - results (dict): Nested dictionary of results keyed by attribute and repair level.
    - protected_attributes (list): List of sensitive attribute names.
    - repair_levels (list): List of repair levels used in the experiment.

    Returns:
    None
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Define metric groups
    perf_order = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    perf_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    fair_order = ['demographic_parity_ratio', 'equalized_odds_ratio',
                  'demographic_parity_difference', 'equalized_odds_difference']
    fair_labels = ['Dem. Parity Ratio', 'Equal. Odds Ratio',
                   'Dem. Parity Diff', 'Equal. Odds Diff']
    
    # Flatten results to DataFrame
    plot_data = []
    for attr in protected_attributes:
        for rl in repair_levels:
            key = f"{attr}_repair_{rl}"
            for fold_metrics in results.get(key, []):
                for metric, value in fold_metrics.items():
                    plot_data.append({
                        'Attribute': attr,
                        'Repair Level': rl,
                        'Metric': metric,
                        'Value': value,
                        'Metric Type': 'Performance' if metric in perf_order else 'Fairness'
                    })
    
    df = pd.DataFrame(plot_data)

    for attr in protected_attributes:
        attr_data = df[df['Attribute'] == attr]
        perf_data_all = attr_data[attr_data['Metric'].isin(perf_order)]
        fair_data_all = attr_data[attr_data['Metric'].isin(fair_order)]

        # Compute y-limits per metric type for this attribute
        perf_ymin, perf_ymax = perf_data_all['Value'].min() - 0.05, perf_data_all['Value'].max() + 0.05
        fair_ymin, fair_ymax = fair_data_all['Value'].min() - 0.05, fair_data_all['Value'].max() + 0.05

        # Create figure
        fig, axes = plt.subplots(2, len(repair_levels), figsize=(30, 16))
        fig.suptitle(f'Metrics for Protected Attribute: {attr}', fontsize=18, y=1.02)

        for j, rl in enumerate(repair_levels):
            rl_data = attr_data[attr_data['Repair Level'] == rl]

            # Performance (top row)
            perf_data = rl_data[rl_data['Metric'].isin(perf_order)]
            perf_palette = {metric: sns.color_palette("Blues", len(perf_order))[i] 
                            for i, metric in enumerate(perf_order)}
            sns.barplot(
                data=perf_data, x='Metric', y='Value', hue='Metric',
                order=perf_order, errorbar='sd', capsize=0.1,
                ax=axes[0, j], palette=perf_palette
            )
            axes[0, j].set_title(f'Repair Level = {rl}', fontsize=14)
            axes[0, j].set_xticks(np.arange(len(perf_order)))
            axes[0, j].set_xticklabels(perf_labels, rotation=45, ha='right', fontsize=12)
            axes[0, j].set_ylim(perf_ymin, perf_ymax)
            axes[0, j].set_xlabel('')
            axes[0, j].set_ylabel('Score', fontsize=12)

            # Add mean ± std value labels
            for k, metric in enumerate(perf_order):
                metric_vals = perf_data[perf_data['Metric'] == metric]['Value']
                mean = metric_vals.mean()
                std = metric_vals.std()
                axes[0, j].text(k, mean + 0.02, f'{mean:.2f}±{std:.2f}', 
                                ha='center', va='bottom', fontsize=10)

            # Fairness (bottom row)
            fair_data = rl_data[rl_data['Metric'].isin(fair_order)]
            fair_palette = {metric: sns.color_palette("Blues_d", len(fair_order))[i] 
                            for i, metric in enumerate(fair_order)}
            sns.barplot(
                data=fair_data, x='Metric', y='Value', hue='Metric',
                order=fair_order, errorbar='sd', capsize=0.1,
                ax=axes[1, j], palette=fair_palette
            )
            axes[1, j].set_xticks(np.arange(len(fair_order)))
            axes[1, j].set_xticklabels(fair_labels, rotation=45, ha='right', fontsize=12)
            axes[1, j].set_ylim(fair_ymin, fair_ymax)
            axes[1, j].set_xlabel('')
            axes[1, j].set_ylabel('Score', fontsize=12)

            for k, metric in enumerate(fair_order):
                metric_vals = fair_data[fair_data['Metric'] == metric]['Value']
                mean = metric_vals.mean()
                std = metric_vals.std()
                axes[1, j].text(k, mean + (0.02 if mean > 0 else -0.02), 
                               f'{mean:.2f}±{std:.2f}', ha='center', 
                               va='bottom' if mean > 0 else 'top', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'metrics_{attr}.png', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()
