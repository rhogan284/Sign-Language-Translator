import csv
from collections import Counter

def count_gloss_values(csv_file):
    gloss_counts = Counter()
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gloss_counts[row['gloss']] += 1
    return gloss_counts

def main():
    csv_file = 'CSVs and JSONs/.csv'
    gloss_counts = count_gloss_values(csv_file)
    sorted_counts = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)
    for gloss, count in sorted_counts:
        print(f"{gloss}: {count}")

if __name__ == "__main__":
    main()
