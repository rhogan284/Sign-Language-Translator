import json
import pandas as pd
import csv

def convert_json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data[0].keys())

        for row in data:
            row_num = 0
            writer.writerow(row.values())
            print("Row " + str(row_num) + " complete")
            row_num += 1


json_file = 'nslt_100.json'
csv_file = '../test.csv'
convert_json_to_csv(json_file, csv_file)

