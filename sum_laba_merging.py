# Обʼєднюємо омоніми:
#   тут ми на основі відфільтрованого дата фрейму sum14_filtered.json
#   формуємо дата фрейм, який об'єднює усі сутності, які мають однакове поле "lemma"

import json

input_file_path = "data/final_results/sum14_filtered.json"
output_file_path = "data/final_results/sum14_merged.json"


def merge_objects(objects):
    merged_objects = {}

    for obj in objects:
        lemma = obj['lemma']

        if lemma not in merged_objects:
            merged_objects[lemma] = {'lemma': lemma, 'synsets': [], 'accent_positions': []}

        merged_objects[lemma]['synsets'].extend(obj.get('synsets', []))
        acc_pos = obj.get('accent_positions', [])
        merged_objects[lemma]['accent_positions'].extend(acc_pos if acc_pos else [])

    return merged_objects


def main():
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = [json.loads(line) for line in input_file]

    # Merge objects based on 'lemma'
    merged_objects = merge_objects(data)

    # Extract values and save them to a new JSON file

    output_data = list(merged_objects.values())
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_data:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    print(f"Number of objects in MERGED file: {len(output_data)}")

if __name__ == '__main__':
    main()