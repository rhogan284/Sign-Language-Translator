import pandas as pd
import ast

df = pd.read_csv('../CSVs and JSONs/Labels.csv')

df['instances'] = df['instances'].apply(ast.literal_eval)

new_rows = []

for index, row in df.iterrows():
    gloss = row['gloss']

    for instance in row['instances']:
        video_id = instance['video_id'].lstrip("'")
        new_row = {'video_id': video_id, 'gloss': gloss}
        new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)

new_df.to_csv('video_ids_per_gloss.csv', index=False)
