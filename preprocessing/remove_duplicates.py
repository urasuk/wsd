import stats as stats_data
import json
import matplotlib.pyplot as plt

train_file_path = stats_data.train_lem_file_path
test_file_path = stats_data.test_lem_file_path

input_file_path = stats_data.pairs_forms_file_path

output_file_path = "data/final_results/sum14_pairs_forms_deduplicated.jsonl"

def calc_number_of_pairs(df):
    n = 0
    for entity in df:
        pairs_list = list(entity.values())[0]
        n+=len(pairs_list)
    return n

def main():
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lemma_with_pairs = [json.loads(line) for line in file]

    print(calc_number_of_pairs(lemma_with_pairs))

    filtered_lemma_with_pairs = []
    deleted_lemmas_strs = []
    for lemma in lemma_with_pairs:
        lemma_str = list(lemma.keys())[0]

        if lemma_str in deleted_lemmas_strs:
            # deleted_lemmas_strs.remove(lemma_str)
            continue

        pairs_list = list(lemma.values())[0]
        synonyms_entity = pairs_list[0]["synonyms"]
        synonyms_strs = [syn["lemma"] for syn in synonyms_entity]

        if len(synonyms_strs) < 2:
            filtered_lemma_with_pairs.append(lemma)
            continue

        synonyms_strs.remove(lemma_str)
        deleted_lemmas_strs.extend(synonyms_strs)

        filtered_lemma_with_pairs.append(lemma)


    
    print(f"number of deleted: {len(deleted_lemmas_strs)}")
    print(f"before lemmas quantity: {len(lemma_with_pairs)}")
    print(f"after deduplicating lemmas quantity: {len(filtered_lemma_with_pairs)}")

    print(calc_number_of_pairs(filtered_lemma_with_pairs))


    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in filtered_lemma_with_pairs:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

if __name__ == '__main__':
    main()


#  2210487
# number of deleted: 7331
# before lemmas quantity: 32613
# after deduplicating lemmas quantity: 25717
# 1384769

# Total number of Positive [✚] pairs: 377685
# Total number of Negative [-] pairs: 1007084

# Train actual: positive🔥 = 339916  (eval: 339917) | negative = 940483
# Test  actual: positive🔥 = 37769  (eval: 37768) | negative = 66601
# Total pairs: 1384769

# 🔥 Train: New number of negative examples: 339916
# 🔥 Test: New number of negative examples: 37769
# Train final: 679832
# Test final: 75538

# (75538/(679832+75538))*100 = 10% 🔥

# [Тренувальний набір] Total number of Positive [✚] pairs: 339916
# [Тренувальний набір] Total number of Negative [-] pairs: 339916
# [Тестувальний набір] Total number of Positive [✚] pairs: 37769
# [Тестувальний набір] Total number of Negative [-] pairs: 37769