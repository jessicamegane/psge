import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional interactive plotting
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import write_html
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def save_interactive_lineplot(df: pd.DataFrame, x: str, y: str, color: str, title: str, out_html: str):
    """Save an interactive Plotly line plot (means across groups)"""
    if not PLOTLY_AVAILABLE:
        return

    # Compute mean per group (config) per x value
    mean_df = df.groupby([x, color], as_index=False)[y].mean()
    fig = px.line(mean_df, x=x, y=y, color=color, title=title)
    fig.update_layout(
        legend=dict(
            title=color,
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
        ),
        margin=dict(t=80, r=200)
    )
    fig.write_html(out_html, include_plotlyjs='cdn')

# Set plot styles
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.facecolor'] = 'FFFFFF'
plt.rcParams['axes.facecolor'] = 'FFFFFF'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({'font.size': 13})

# Column names from the progress_report.csv
column_names = [
    'generation', 'best_fitness', 'best_gen_fitness', 'mean_fitness_population', 'std_fitness_population',
    'best_test', 'mean_test_population', 'std_test_population', 'best_depth', 'mean_depth_population',
    'std_depth_population', 'length_best_genotype',
    'percentage_unique_individuals'
]

def main():
    base_dir = "/media/storage/jessica/search_strategy/full_experiments_fix/"
    vis_dir = "exp_visualization_fix"
    os.makedirs(vis_dir, exist_ok=True)

    # Collect unique benchmarks by scanning the directory structure
    benchmarks = set()
    for sub_exp in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, sub_exp)
        if os.path.isdir(sub_path):
            for item in os.listdir(sub_path):
                item_path = os.path.join(sub_path, item)
                if os.path.isdir(item_path):
                    benchmarks.add(item)
    benchmarks = list(benchmarks)
    print(f"Found benchmarks: {benchmarks}")

    for benchmark in benchmarks:
        if benchmark != 'nguyen5polynomial':
            print(f"Skipping benchmark: {benchmark} (no standard configuration)")
            continue
        bench_vis_dir = os.path.join(vis_dir, benchmark)
        os.makedirs(bench_vis_dir, exist_ok=True)

        # Collect data from all configurations, including standard
        data = []
        for sub_exp in os.listdir(base_dir):
            sub_path = os.path.join(base_dir, sub_exp)
            if not os.path.isdir(sub_path):
                continue

            bench_path = os.path.join(sub_path, benchmark)
            if not os.path.isdir(bench_path):
                continue
            
            # if not sub_path.endswith('no_elite'):
            #     continue

            if sub_exp == 'standard':
                config_prefix = 'standard'
            else:
                # match = re.match(r'eda_remap_nbest(\d+)', sub_exp)
                # if not match:
                #     continue
                # n_best = match.group(1)
                # config_prefix = f'eda_nbest{n_best}'
                config_prefix = sub_exp

            # Now loop over potential learning factor folders
            for lf_folder in os.listdir(bench_path):
                lf_path = os.path.join(bench_path, lf_folder)
                if not os.path.isdir(lf_path):
                    continue
                try:
                    learning_factor = float(lf_folder)
                except ValueError:
                    continue
                config = f'{config_prefix}_lf{learning_factor}'
                for run_folder in os.listdir(lf_path):
                    if not run_folder.startswith('run_'):
                        continue
                    run_path = os.path.join(lf_path, run_folder)
                    file_path = os.path.join(run_path, 'progress_report.csv')
                    if not os.path.exists(file_path):
                        continue
                    df = pd.read_csv(file_path, sep='\t', header=None, names=column_names, na_values="nan")
                    run_num = re.search(r'\d+', run_folder).group()
                    df['run'] = run_num
                    df['config'] = config
                    data.append(df)

        if not data:
            print(f"No data found for benchmark: {benchmark}")
            continue

        all_df = pd.concat(data, ignore_index=True)
        all_df.to_csv(os.path.join(bench_vis_dir, 'aggregated_data.csv'), sep='\t', index=False, na_rep="nan")
        print(f"Aggregated data for {benchmark} saved to {bench_vis_dir}")

        # Define metrics for plotting
        metrics = [
            ('best_fitness', 'Mean Error of Best Individual'),
            ('mean_fitness_population', 'Population Mean Error'),
            ('best_depth', 'Best Individual Depth'),
            ('mean_depth_population', 'Population Mean Depth'),
            ('length_best_genotype', 'Best Individual Genotype Length'),
            ('percentage_unique_individuals', 'Percentage of Unique individuals'),
        ]

        # Generate line plots for each metric, with all configs including standard
        for y_col, title in metrics:
            fig, ax = plt.subplots()
            # if title == "Mean_fitness_population" logarithmic scale for better visibility
            if y_col == "mean_fitness_population":
                ax.set_yscale('log')
            sns.lineplot(
                data=all_df, x='generation', y=y_col, hue='config',
                estimator='mean', errorbar='sd', ax=ax
            )
            ax.set_title(f'{title} for {benchmark}')
            ax.set_xlabel('Generation')
            ax.set_ylabel(title)
            plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, f'{y_col}_vs_generation.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot: {save_path}")

            # Interactive version
            html_path = os.path.join(bench_vis_dir, f'{y_col}_vs_generation.html')
            save_interactive_lineplot(all_df, 'generation', y_col, 'config', f'{title} for {benchmark}', html_path)
            if PLOTLY_AVAILABLE:
                print(f"Saved interactive plot: {html_path}")

        # Additional helpful plots
        # 1. Population Fitness Std (mean across runs)
        fig, ax = plt.subplots()
        sns.lineplot(
            data=all_df, x='generation', y='std_fitness_population', hue='config',
            estimator='mean', errorbar='sd', ax=ax
        )
        ax.set_title(f'Population Error Std for {benchmark}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Error Std')
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'pop_fitness_std_vs_generation.png')
        plt.savefig(save_path)
        plt.close()

        html_path = os.path.join(bench_vis_dir, 'pop_fitness_std_vs_generation.html')
        save_interactive_lineplot(all_df, 'generation', 'std_fitness_population', 'config', f'Population Error Std for {benchmark}', html_path)
        if PLOTLY_AVAILABLE:
            print(f"Saved interactive plot: {html_path}")

        # 2. Best Gen Fitness
        fig, ax = plt.subplots()
        sns.lineplot(
            data=all_df, x='generation', y='best_gen_fitness', hue='config',
            estimator='mean', errorbar='sd', ax=ax
        )
        ax.set_title(f'Best Gen Error for {benchmark}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Gen Error')
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'best_gen_fitness_vs_generation.png')
        plt.savefig(save_path)
        plt.close()

        html_path = os.path.join(bench_vis_dir, 'best_gen_fitness_vs_generation.html')
        save_interactive_lineplot(all_df, 'generation', 'best_gen_fitness', 'config', f'Best Gen Error for {benchmark}', html_path)
        if PLOTLY_AVAILABLE:
            print(f"Saved interactive plot: {html_path}")

        # 3. Best Test (if data available)
        if 'best_test' in all_df.columns and all_df['best_test'].notna().any():
            fig, ax = plt.subplots()
            sns.lineplot(
                data=all_df, x='generation', y='best_test', hue='config',
                estimator='mean', errorbar='sd', ax=ax
            )
            ax.set_title(f'Best Test for {benchmark}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Best Test')
            plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, 'best_test_vs_generation.png')
            plt.savefig(save_path)
            plt.close()

            html_path = os.path.join(bench_vis_dir, 'best_test_vs_generation.html')
            save_interactive_lineplot(all_df, 'generation', 'best_test', 'config', f'Best Test for {benchmark}', html_path)
            if PLOTLY_AVAILABLE:
                print(f"Saved interactive plot: {html_path}")

        # 4. Population Depth Std
        fig, ax = plt.subplots()
        sns.lineplot(
            data=all_df, x='generation', y='std_depth_population', hue='config',
            estimator='mean', errorbar='sd', ax=ax
        )
        ax.set_title(f'Population Depth Std for {benchmark}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Depth Std')
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'pop_depth_std_vs_generation.png')
        plt.savefig(save_path)
        plt.close()

        html_path = os.path.join(bench_vis_dir, 'pop_depth_std_vs_generation.html')
        save_interactive_lineplot(all_df, 'generation', 'std_depth_population', 'config', f'Population Depth Std for {benchmark}', html_path)
        if PLOTLY_AVAILABLE:
            print(f"Saved interactive plot: {html_path}")

        # Boxplot at final generation for best_fitness
        max_gen = all_df['generation'].max()
        final_df = all_df[all_df['generation'] == max_gen]
        if not final_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=final_df, x='config', y='best_fitness', ax=ax)
            ax.set_title(f'Boxplot of Best Error at Generation {max_gen} for {benchmark}')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Best Error')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, 'boxplot_best_fitness_final.png')
            plt.savefig(save_path)
            plt.close()

        # Boxplot at final generation for best_test (if available)
        if 'best_test' in all_df.columns and final_df['best_test'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=final_df, x='config', y='best_test', ax=ax)
            ax.set_title(f'Boxplot of Best Test at Generation {max_gen} for {benchmark}')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Best Test')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, 'boxplot_best_test_final.png')
            plt.savefig(save_path)
            plt.close()

        # Boxplot at final generation for percentage_unique_individuals
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=final_df, x='config', y='percentage_unique_individuals', ax=ax)
        ax.set_title(f'Boxplot of Percentage Unique Individuals at Generation {max_gen} for {benchmark}')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Unique Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'boxplot_unique_percentage_final.png')
        plt.savefig(save_path)
        plt.close()

        # Boxplot at final generation for best individual of generation (best_gen_fitness)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=final_df, x='config', y='best_gen_fitness', ax=ax)
        ax.set_title(f'Boxplot of Best Individual of Generation {max_gen} for {benchmark}')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Best Error')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'boxplot_best_individual_gen_final.png')
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    main()