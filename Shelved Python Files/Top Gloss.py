import pandas as pd

data = pd.read_csv('../CSVs and JSONs/Video_ids_Final.csv')

data['video_id'] = data['video_id'].astype(str).str.zfill(5)

gloss_counts = data['gloss'].value_counts().reset_index()
gloss_counts.columns = ['gloss', 'count']

gloss_counts_sorted = gloss_counts.sort_values(by='count', ascending=False)

top_50_glosses = gloss_counts_sorted.head(50)
top_100_glosses = gloss_counts_sorted.head(100)

# Filter the original data for the top 50 and 100 glosses
top_50_videos = data[data['gloss'].isin(top_50_glosses['gloss'])]
top_100_videos = data[data['gloss'].isin(top_100_glosses['gloss'])]

# Save the filtered data to new CSV files
top_50_videos.to_csv('CSVs and JSONs/top_50_glosses.csv', index=False)
top_100_videos.to_csv('CSVs and JSONs/top_100_glosses.csv', index=False)

print(top_50_videos.head())