import pandas as pd
import ast

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Labels.csv')

# Convert string representation of list to actual list of dictionaries
df['instances'] = df['instances'].apply(ast.literal_eval)

# Create an empty list to hold the rows of the new DataFrame
new_rows = []

# Iterate over each row of the DataFrame
for index, row in df.iterrows():
    # Extract gloss value
    gloss = row['gloss']
    # Iterate over each instance in the 'instances' column
    for instance in row['instances']:
        # Remove the single quote from the beginning of the video ID, if present
        video_id = instance['video_id'].lstrip("'")
        # Create a new row with 'video_id' and 'gloss' columns
        new_row = {'video_id': video_id, 'gloss': gloss}
        # Append the new row to the list
        new_rows.append(new_row)

# Create a new DataFrame from the list of rows
new_df = pd.DataFrame(new_rows)

# Display the new DataFrame
print(new_df)

# Save the new DataFrame to a new CSV file
new_df.to_csv('video_ids_per_gloss.csv', index=False)
