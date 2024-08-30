import os
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
output_dir = '/home/vtiyyal1/data-mdredze1/vtiyyal1/cannabis_project/outputs'

# Initialize statistics tracking
statistics = []

# Process each annotated CSV file
for filename in os.listdir(output_dir):
    if filename.startswith('annotated_') and filename.endswith('.csv'):
        file_path = os.path.join(output_dir, filename)
        try:
            # Load annotated data
            df = pd.read_csv(file_path)
            
            # Gather statistics
            num_instances = len(df)
            num_adverse_events = df['roberta_ae_prediction'].sum()
            subreddit = filename.replace('annotated_', '').replace('_filtered_data.csv', '')
            statistics.append({'subreddit': subreddit, 'num_instances': num_instances, 'num_adverse_events': num_adverse_events})

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save statistics to a CSV file
statistics_df = pd.DataFrame(statistics)
statistics_df.to_csv(os.path.join(output_dir, 'annotation_statistics.csv'), index=False)

# Sample and save some examples
sampled_df = pd.DataFrame()
for filename in statistics_df['subreddit']:
    df = pd.read_csv(os.path.join(output_dir, f'annotated_{filename}_filtered_data.csv'))
    sampled_df = pd.concat([sampled_df, df.sample(min(10, len(df)))], ignore_index=True)

sampled_df.to_csv(os.path.join(output_dir, 'sampled_annotations.csv'), index=False)

# Generate plots
def plot_statistics(statistics_df, output_dir):
    # Plot number of instances per subreddit
    plt.figure(figsize=(12, 6))
    plt.bar(statistics_df['subreddit'], statistics_df['num_instances'], color='skyblue')
    plt.xlabel('Subreddit')
    plt.ylabel('Number of Instances')
    plt.title('Number of Instances per Subreddit')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'num_instances_per_subreddit.png'))
    plt.close()

    # Plot number of adverse events per subreddit
    plt.figure(figsize=(12, 6))
    plt.bar(statistics_df['subreddit'], statistics_df['num_adverse_events'], color='salmon')
    plt.xlabel('Subreddit')
    plt.ylabel('Number of Adverse Events')
    plt.title('Number of Adverse Events per Subreddit')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'num_adverse_events_per_subreddit.png'))
    plt.close()

    # Plot percentage of adverse events per subreddit
    statistics_df['percent_adverse_events'] = (statistics_df['num_adverse_events'] / statistics_df['num_instances']) * 100
    plt.figure(figsize=(12, 6))
    plt.bar(statistics_df['subreddit'], statistics_df['percent_adverse_events'], color='lightgreen')
    plt.xlabel('Subreddit')
    plt.ylabel('Percentage of Adverse Events')
    plt.title('Percentage of Adverse Events per Subreddit')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percent_adverse_events_per_subreddit.png'))
    plt.close()

plot_statistics(statistics_df, output_dir)
