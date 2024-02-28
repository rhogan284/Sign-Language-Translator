import json
import pandas as pd
import csv


def convert_json_to_csv(json_file, csv_file):
    # Open the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header using the keys from the JSON data
        writer.writerow(data[0].keys())

        # Write the data
        for row in data:
            row_num = 0
            writer.writerow(row.values())
            print("Row " + str(row_num) + " complete")
            row_num += 1


# Example usage
json_file = 'WLASL_v0.3.json'
csv_file = 'Labels.csv'
convert_json_to_csv(json_file, csv_file)
