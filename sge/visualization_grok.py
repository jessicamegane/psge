import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
 
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
    'percentage_unique_individuals', 'percentage_new_individuals'
]
 
from plotly.offline import plot
import plotly.express as px


def get_line_style(config):
    """Determine line style based on config name prefix (tuples for seaborn dashes)"""
    if config.startswith('sgef'):
        return ()  # solid line
    elif config.startswith('psge'):
        return (1, 1)  # dotted line
    elif config.startswith('copsge'):
        return (2, 2)  # dashed line
    else:
        return ()  # solid line


def get_line_dash(config):
    """Determine dash pattern for plotly based on config name prefix"""
    if config.startswith('sgef'):
        return 'solid'
    elif config.startswith('psge'):
        return 'dot'
    elif config.startswith('copsge'):
        return 'dash'
    else:
        return 'solid'


def save_plotly_lineplot(df, x, y, color, title, save_path, yaxis_range=None, log_scale=False, show_std=False, line_dash_map=None):
    # Aggregate by generation and config to get mean
    if show_std:
        grouped = df.groupby([x, color]).agg({y: ['mean', 'std']}).reset_index()
        grouped.columns = [x, color, f'{y}_mean', f'{y}_std']
        fig = px.line(grouped, x=x, y=f'{y}_mean', color=color, title=title, error_y=f'{y}_std')
    else:
        grouped = df.groupby([x, color]).agg({y: 'mean'}).reset_index()
        fig = px.line(grouped, x=x, y=y, color=color, title=title)
    
    # Apply line dash patterns if mapping provided
    if line_dash_map:
        for trace in fig.data:
            config_name = trace.name
            if config_name in line_dash_map:
                trace.line.dash = line_dash_map[config_name]
    
    layout_dict = dict(
        legend=dict(
            orientation='v',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(l=40, r=200, t=60, b=40),
        xaxis_title=x,
        yaxis_title=y
    )
    
    if yaxis_range:
        layout_dict['yaxis'] = dict(range=yaxis_range)
    elif log_scale:
        layout_dict['yaxis'] = dict(type='log')
    
    fig.update_layout(**layout_dict)
    plot(fig, filename=save_path, auto_open=False)


def save_plotly_boxplot(df, x, y, title, save_path, category_order=None, color_discrete_map=None):
    fig = px.box(df, x=x, y=y, color=x, title=title, category_orders={x: category_order} if category_order else None, color_discrete_map=color_discrete_map)
    fig.update_layout(
        legend=dict(
            orientation='v',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(l=40, r=200, t=60, b=40),
        xaxis_title=x,
        yaxis_title=y
    )
    plot(fig, filename=save_path, auto_open=False)



def main(show_std=True):
    base_dir = "/Users/jessicamegane/Downloads/fix_repo/hybrid psge copsge/psge/sge/experiments_500_gen/"
    vis_dir = "exp_visualization_500_gen"
    os.makedirs(vis_dir, exist_ok=True)

 
    # Collect unique benchmarks by scanning the directory structure
    benchmarks = set()
    for sub_exp in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, sub_exp)
        # if "cma_es" in sub_path:
        #     continue
        if os.path.isdir(sub_path):
            for item in os.listdir(sub_path):
                item_path = os.path.join(sub_path, item)
                if os.path.isdir(item_path):
                    benchmarks.add(item)
    benchmarks = list(benchmarks)
    print(f"Found benchmarks: {benchmarks}")
 
    for benchmark in benchmarks:
        # if benchmark != 'nguyen5polynomial':  # Only visualize nguyen5polynomial for now
        #     print(f"Skipping benchmark: {benchmark} (no standard configuration)")
        #     continue
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
 
            # Now loop over potential learning factor folders OR direct run folders
            for entry in os.listdir(bench_path):
                entry_path = os.path.join(bench_path, entry)
                if not os.path.isdir(entry_path):
                    continue

                # Direct run folder case: bench_path/run_X
                if entry.startswith('run_'):
                    config = config_prefix
                    run_folder = entry
                    run_path = entry_path
                    file_path = os.path.join(run_path, 'progress_report.csv')
                    if not os.path.exists(file_path):
                        continue
                    df = pd.read_csv(file_path, sep='\t', header=None, names=column_names, na_values="nan")
                    run_num = re.search(r'\d+', run_folder).group()
                    df['run'] = run_num
                    df['config'] = config
                    data.append(df)
                    continue

                # Learning factor subfolder case: bench_path/<lf>/run_X
                try:
                    learning_factor = float(entry)
                except ValueError:
                    continue

                config = f'{config_prefix}_lf{learning_factor}'
                lf_path = entry_path
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
        
        # Establish consistent config order and palette for all plots
        config_order = all_df['config'].unique().tolist()
        palette = sns.color_palette("husl", len(config_order))
        color_map = {config: palette[i] for i, config in enumerate(config_order)}
        # Create color_discrete_map for plotly (convert RGB tuples to hex)
        import matplotlib.colors as mcolors
        color_discrete_map = {config: mcolors.to_hex(palette[i]) for i, config in enumerate(config_order)}
        # Create line style maps based on config prefix
        line_style_map = {config: get_line_style(config) for config in config_order}
        line_dash_map = {config: get_line_dash(config) for config in config_order}
 
        # Define metrics for plotting
        metrics = [
            ('best_fitness', 'Mean Error of Best Individual'),
            ('mean_fitness_population', 'Population Mean Error'),
            ('best_depth', 'Best Individual Depth'),
            ('mean_depth_population', 'Population Mean Depth'),
            ('length_best_genotype', 'Best Individual Genotype Length'),
            ('percentage_unique_individuals', 'Percentage of Unique individuals'),
            ('percentage_new_individuals', 'Percentage of New individuals'),
        ]
 
        # Generate line plots for each metric, with all configs including standard
        for y_col, title in metrics:
            fig, ax = plt.subplots()
            # if title == "Mean_fitness_population" logarithmic scale for better visibility
            if y_col == "mean_fitness_population":
                ax.set_yscale('log')
            sns.lineplot(
                data=all_df, x='generation', y=y_col, hue='config', style='config',
                estimator='mean', errorbar=('sd' if show_std else None), ax=ax,
                palette=color_map, dashes=line_style_map, hue_order=config_order, style_order=config_order
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

            # Plotly version
            p_save_path = os.path.join(bench_vis_dir, f'{y_col}_vs_generation.html')
            yaxis_range = [0, 100] if 'percentage' in y_col else None
            log_scale = y_col == "mean_fitness_population"
            save_plotly_lineplot(all_df, x='generation', y=y_col, color='config', title=f'{title} for {benchmark}', save_path=p_save_path, yaxis_range=yaxis_range, log_scale=log_scale, show_std=show_std, line_dash_map=line_dash_map)
            print(f"Saved interactive plot: {p_save_path}")
 
        # Additional helpful plots
        # 1. Population Fitness Std (mean across runs)
        fig, ax = plt.subplots()
        sns.lineplot(
            data=all_df, x='generation', y='std_fitness_population', hue='config', style='config',
            estimator='mean', errorbar='sd', ax=ax,
            palette=color_map, dashes=line_style_map, hue_order=config_order, style_order=config_order
        )
        ax.set_title(f'Population Error Std for {benchmark}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Error Std')
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'pop_fitness_std_vs_generation.png')
        plt.savefig(save_path)
        plt.close()
        p_save_path = os.path.join(bench_vis_dir, 'pop_fitness_std_vs_generation.html')
        save_plotly_lineplot(all_df, x='generation', y='std_fitness_population', color='config', title=f'Population Error Std for {benchmark}', save_path=p_save_path, yaxis_range=None, show_std=show_std, line_dash_map=line_dash_map)
        print(f"Saved interactive plot: {p_save_path}")
 
        # 2. Best Gen Fitness
        fig, ax = plt.subplots()
        sns.lineplot(
            data=all_df, x='generation', y='best_gen_fitness', hue='config', style='config',
            estimator='mean', errorbar='sd', ax=ax,
            palette=color_map, dashes=line_style_map, hue_order=config_order, style_order=config_order
        )
        ax.set_title(f'Best Gen Error for {benchmark}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Gen Error')
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'best_gen_fitness_vs_generation.png')
        plt.savefig(save_path)
        plt.close()
        p_save_path = os.path.join(bench_vis_dir, 'best_gen_fitness_vs_generation.html')
        save_plotly_lineplot(all_df, x='generation', y='best_gen_fitness', color='config', title=f'Best Gen Error for {benchmark}', save_path=p_save_path, yaxis_range=None, show_std=show_std, line_dash_map=line_dash_map)
        print(f"Saved interactive plot: {p_save_path}")
 
        # 3. Best Test (if data available)
        if 'best_test' in all_df.columns and all_df['best_test'].notna().any():
            fig, ax = plt.subplots()
            sns.lineplot(
                data=all_df, x='generation', y='best_test', hue='config', style='config',
                estimator='mean', errorbar='sd', ax=ax,
                palette=color_map, dashes=line_style_map, hue_order=config_order, style_order=config_order
            )
            ax.set_title(f'Best Test for {benchmark}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Best Test')
            plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, 'best_test_vs_generation.png')
            plt.savefig(save_path)
            plt.close()
            p_save_path = os.path.join(bench_vis_dir, 'best_test_vs_generation.html')
            save_plotly_lineplot(all_df, x='generation', y='best_test', color='config', title=f'Best Test for {benchmark}', save_path=p_save_path, yaxis_range=None, show_std=show_std, line_dash_map=line_dash_map)
            print(f"Saved interactive plot: {p_save_path}")
 
        # 4. Population Depth Std
        fig, ax = plt.subplots()
        sns.lineplot(
            data=all_df, x='generation', y='std_depth_population', hue='config', style='config',
            estimator='mean', errorbar='sd', ax=ax,
            palette=color_map, dashes=line_style_map, hue_order=config_order, style_order=config_order
        )
        ax.set_title(f'Population Depth Std for {benchmark}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Depth Std')
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'pop_depth_std_vs_generation.png')
        plt.savefig(save_path)
        plt.close()
        p_save_path = os.path.join(bench_vis_dir, 'pop_depth_std_vs_generation.html')
        save_plotly_lineplot(all_df, x='generation', y='std_depth_population', color='config', title=f'Population Depth Std for {benchmark}', save_path=p_save_path, yaxis_range=None, show_std=show_std, line_dash_map=line_dash_map)
        print(f"Saved interactive plot: {p_save_path}")
 
        # Boxplot at final generation for best_fitness
        max_gen = all_df['generation'].max()
        final_df = all_df[all_df['generation'] == max_gen]
        if not final_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=final_df, x='config', y='best_fitness', ax=ax, order=config_order, palette=color_map)
            ax.set_title(f'Boxplot of Best Error at Generation {max_gen} for {benchmark}')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Best Error')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, 'boxplot_best_fitness_final.png')
            plt.savefig(save_path)
            plt.close()
            p_save_path = os.path.join(bench_vis_dir, 'boxplot_best_fitness_final.html')
            save_plotly_boxplot(final_df, x='config', y='best_fitness', title=f'Boxplot of Best Error at Generation {max_gen} for {benchmark}', save_path=p_save_path, category_order=config_order, color_discrete_map=color_discrete_map)
            print(f"Saved interactive plot: {p_save_path}")
 
        # Boxplot at final generation for best_test (if available)
        if 'best_test' in all_df.columns and final_df['best_test'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=final_df, x='config', y='best_test', ax=ax, order=config_order, palette=color_map)
            ax.set_title(f'Boxplot of Best Test at Generation {max_gen} for {benchmark}')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Best Test')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = os.path.join(bench_vis_dir, 'boxplot_best_test_final.png')
            plt.savefig(save_path)
            plt.close()
            p_save_path = os.path.join(bench_vis_dir, 'boxplot_best_test_final.html')
            save_plotly_boxplot(final_df, x='config', y='best_test', title=f'Boxplot of Best Test at Generation {max_gen} for {benchmark}', save_path=p_save_path, category_order=config_order, color_discrete_map=color_discrete_map)
            print(f"Saved interactive plot: {p_save_path}")
 
        # Boxplot at final generation for percentage_unique_individuals
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=final_df, x='config', y='percentage_unique_individuals', ax=ax, order=config_order, palette=color_map)
        ax.set_title(f'Boxplot of Percentage Unique Individuals at Generation {max_gen} for {benchmark}')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Unique Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(bench_vis_dir, 'boxplot_unique_percentage_final.png')
        plt.savefig(save_path)
        plt.close()
        p_save_path = os.path.join(bench_vis_dir, 'boxplot_unique_percentage_final.html')
        save_plotly_boxplot(final_df, x='config', y='percentage_unique_individuals', title=f'Boxplot of Percentage Unique Individuals at Generation {max_gen} for {benchmark}', save_path=p_save_path, category_order=config_order, color_discrete_map=color_discrete_map)
        print(f"Saved interactive plot: {p_save_path}")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for experiments.")
    parser.add_argument('--no-std', action='store_true', help="Disable standard deviation error bars in plots (default: enabled)")
    args = parser.parse_args()
    
    show_std = not args.no_std
    main(show_std=show_std)