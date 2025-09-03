import csv
import sys

def count_numbers(input_csv, output_csv):
    # initialize counts for 0..100
    counts = [0] * 101

    with open(input_csv, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                num = int(row[0])
                if 0 <= num <= 100:
                    counts[num] += 1
            except ValueError:
                # skip invalid entries
                continue

    # write results to output csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Number", "Count"])
        for i, c in enumerate(counts):
            writer.writerow([i, c])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python count_numbers.py <input_csv> <output_csv>")
        sys.exit(1)

    count_numbers(sys.argv[1], sys.argv[2])
