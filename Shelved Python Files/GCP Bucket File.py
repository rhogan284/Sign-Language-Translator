import pandas as pd

# Load the uploaded CSV file to inspect its contents
file_path = '../CSVs and JSONs/top_100_glosses.csv'
data = pd.read_csv(file_path)

data['START'] = 0
data['END'] = 'inf'

# Format the 'gs://sign-language-videos/video_id' as requested
data['video_id'] = data['video_id'].astype(str).str.zfill(5)
data['video_id'] = data['video_id'].apply(lambda x: f'gs://sign-language-videos-100/Top 100 Videos Processed/{x}_processed.mp4')

# Reorder the columns to match the requested format
data = data[['video_id', 'gloss', 'START', 'END']]

# Save to new CSV file
output_path = '../CSVs and JSONs/formatted_sign_language_videos_100_2.csv'
data.to_csv(output_path, index=False)
