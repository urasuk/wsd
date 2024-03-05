import stats as stats_data
import json
import numpy as np
from itertools import combinations


input_file_path = stats_data.merged_file_path
output_file_path = stats_data.pairs_file_path

def get_unique(original_list):
    seen = []
    unique_list = [item for item in original_list if item not in seen and not seen.append(item)]
    return unique_list

def create_siamese_dataset(data):
    all_examples = []
    for entry in data:
        entry_examples = []
        for sens_id, synset in enumerate(entry['synsets']):
            for example in synset['examples']:
                entry_examples.append({"example": example["ex_text"],
                                       "sense_id": sens_id, 
                                       "lemma": entry["lemma"],
                                       "synonyms": get_unique(entry["synonyms"]) }) # !!!
        all_examples.append(entry_examples)

    dataset = []
    for examples in all_examples:
        pairs = []
        # Iterate through all combinations of pairs
        for pair in combinations(examples, 2):
            sen1 = pair[0]["example"]
            sen2 = pair[1]["example"]
            label = 1 if pair[0]["sense_id"] == pair[1]["sense_id"] else 0
            pairs.append({"sentence1": sen1,
                          "sentence2": sen2, 
                          "label": label, 
                          "lemma": pair[0]["lemma"], 
                          "synonyms": pair[0]["synonyms"]})  # !!!
        # dataset.extend(pairs)
        dataset.append({examples[0]["lemma"]: pairs}) #% 2024-03-03: Proper train/test split (по лемах)

    return dataset


def main():
    # Read JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    # Create Siamese dataset
    siamese_dataset = create_siamese_dataset(data)

    # Save Siamese dataset to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in siamese_dataset:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    # print(f"Paraprases dataset length: {len(siamese_dataset)}")
    #% 2024-03-03: Proper train/test split (по лемах)
    lema_pair_num = [len(d[key]) for d in siamese_dataset for key in d if isinstance(d[key], list)]
    print(f"Paraprases dataset length: {np.array(lema_pair_num).sum()}")

    print(f"Siamese dataset saved to {output_file_path}")



if __name__ == '__main__':
    main()
