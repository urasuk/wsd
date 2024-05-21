# Обʼєднюємо омоніми:
#   тут ми на основі відфільтрованого дата фрейму sum14_filtered.json
#   формуємо дата фрейм, який об'єднює усі сутності, які мають однакове поле "lemma"

import json
import stats as stats_data


# input_file_path = stats_data.filtered_file_path
# input_file_path = "data/final_results/clust_sum14_filtered-extra.json"
input_file_path = "data/final_results/clust_sum14_with_references.json"


output_file_path = "data/final_results/clustering_sum14_merged.json"



def find_acute_accent_positions(input_string):
    positions = []

    if len(positions)!=0:
        print("stop")

    for idx, char in enumerate(input_string):
        if char == '́':
            positions.append(idx-1) # cuz idx - is index of acute symbol
    return positions


def remove_acute_accents(input_string, accent_positions=None):
    accent_positions = accent_positions if accent_positions else find_acute_accent_positions(input_string)

    for i, position in enumerate(accent_positions):
        position -= i
        input_string = input_string[:position + 1] + input_string[position + 2:]

    return input_string

def merge_objects(objects):
    merged_objects = {}

    for obj in objects:
        lemma = obj['lemma']

        if lemma not in merged_objects:
            merged_objects[lemma] = {'lemma': lemma, 'synsets': [], 'accent_positions': [], 'synonyms': [],
                                     'omonim_count': 0}
            synsets_to_add = obj.get('synsets', [])
            synsets_to_add = [{**synset_dict, "index": 0} for synset_dict in synsets_to_add]
        else:
            # merged_objects[lemma][""]
            merged_objects[lemma]["omonim_count"] +=1
            synsets_to_add = obj.get('synsets', [])
            synsets_to_add = [{**synset_dict, "index": merged_objects[lemma]["omonim_count"]}
                               for synset_dict in synsets_to_add]
            

        merged_objects[lemma]['synsets'].extend(synsets_to_add)
        acc_pos = obj.get('accent_positions', [])
        merged_objects[lemma]['accent_positions'].extend(acc_pos if acc_pos else [])
        merged_objects[lemma]['synonyms'].extend(obj.get('synonims', [])) # !!!

        for synset in merged_objects[lemma]['synsets']:
            gloss_no_accents = remove_acute_accents(synset["gloss"])
            synset["gloss"] = gloss_no_accents

    return merged_objects


def main():
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = [json.loads(line) for line in input_file]

    # Merge objects based on 'lemma'
    merged_objects = merge_objects(data)

    # Extract values and save them to a new JSON file

    output_data = list(merged_objects.values())

    # Є такі омоніми (наприклад АВІЗНИЙ), які для 2-го prime сенсу не мають прикладів,
    # а отже 2 омонім не додається в датасет - і датасет має лему з 1 сенсом
    # тому треба зробити фінальний фільтр
    output_data = [entry for entry in output_data if len(entry['synsets']) != 1]
    # before: 34169
    # after:  33884 (33887 - new quantity after fixing апострофи в sum_laba_filtering.py)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_data:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    print(f"Number of objects in MERGED file: {len(output_data)}")

if __name__ == '__main__':
    main()