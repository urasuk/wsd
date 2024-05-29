# import stats as stats_data
import json
# import numpy as np

# input_file_path = stats_data.pairs_forms_file_path

# def main():
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         lines = [json.loads(line) for line in file]

#     for entry1 in lines:
#             for entry2 in lines:
#                 if entry1["sentence1"] == entry1["sentence1"]

# if __name__ == '__main__':
#     main()

import stats as stats_data
import jsonlines

input_file_path = stats_data.pairs_forms_file_path

# Відкриття JSONL файлу для зчитування
# with jsonlines.open(input_file_path) as reader:

with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = [json.loads(line) for line in file]
    
# Створення списку для зберігання дублікатів
duplicates = []
# Створення множини для відстеження вже перевірених рядків
checked_pairs = set()

# Перебір усіх рядків у файлі
for i, entry1 in enumerate(lines):
    for entry2 in lines[i+1:]:
        # Перевірка наявності дублікатів
        if ((entry1["sentence1"] == entry2["sentence1"] and entry1["sentence2"] == entry2["sentence2"]) or
            (entry1["sentence1"] == entry2["sentence2"] and entry1["sentence2"] == entry2["sentence1"])):
            # Якщо пара рядків має дублікати, перевіряємо, чи вже перевіряли їх
            pair = frozenset((entry1["sentence1"], entry1["sentence2"]))
            if pair not in checked_pairs:
                duplicates.append((entry1, entry2))
                # Додаємо пару у відстежувані
                checked_pairs.add(pair)

# Виведення знайдених дублікатів
for entry1, entry2 in duplicates:
    print("Duplicate pair found:")
    print(entry1)
    print(entry2)
