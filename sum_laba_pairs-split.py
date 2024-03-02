import json
import random
import stats as stats_data

input_file_path = stats_data.pairs_file_path
train_file_path = stats_data.train_file_path
test_file_path = stats_data.test_file_path

# Set the seed for reproducibility
random.seed(42)

def split_jsonl(input_file, train_file, test_file, split_ratio=0.75):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the split index
    split_index = int(len(lines) * split_ratio)

    # Split the lines into train and test sets
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]
    print(f"Number of entries in Train: {len(train_lines)}")
    print(f"Number of entries in Test: {len(test_lines)}")

    # Write the train set to a new file
    # with open(train_file, 'w', encoding='utf-8') as train_output_file:
    #     train_output_file.writelines(train_lines)

    # # Write the test set to a new file
    # with open(test_file, 'w', encoding='utf-8') as test_output_file:
    #     test_output_file.writelines(test_lines)


def main():
    # Split the JSONL file
    split_jsonl(input_file_path, train_file_path, test_file_path)


if __name__ == '__main__':
    main()
