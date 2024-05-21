import json
import stats

train_good_file_path = stats.train_lem_file_path
test_good_file_path = stats.test_lem_file_path

# Функція для отримання підстрічки з заданими індексами
def extract_substring(sentence, start_idx, end_idx):
    return sentence[start_idx:end_idx+1]

def show_targets_from_sentences(sentence1, sentence2, entry, substring1, substring2):
    print(f"Sentence #1: {sentence1}")
    print(f"Sentence #2: {sentence2}")
    print(f"LEMMA: {entry['lemma']} |{substring1}|{substring2}|\n")

def main():
    with open(train_good_file_path, "r", encoding="utf-8") as f:
        data_train = [json.loads(line) for line in f]
    with open(test_good_file_path, "r", encoding="utf-8") as f:
        data_test = [json.loads(line) for line in f]


    # Зчитування даних з JSON та виконання операцій
    for data in [data_train, data_test]:
        for i, entry in enumerate(data):
            sentence1 = entry["sentence1"]
            sentence2 = entry["sentence2"]
            sent1_start_idx = entry["sent1_target_pos"]["target_start_idx"]
            sent1_end_idx = entry["sent1_target_pos"]["target_end_idx"]
            sent2_start_idx = entry["sent2_target_pos"]["target_start_idx"]
            sent2_end_idx = entry["sent2_target_pos"]["target_end_idx"]

            substring1 = extract_substring(sentence1, sent1_start_idx, sent1_end_idx)
            substring2 = extract_substring(sentence2, sent2_start_idx, sent2_end_idx)

            # if i < 50: 
                # show_targets_from_sentences(sentence1, sentence2, entry, substring1, substring2)
            
            # TARGETS has no SPACES !!!
            if " " in substring1 or " " in substring2:
                show_targets_from_sentences(sentence1, sentence2, entry, substring1, substring2)



if __name__ == '__main__':
    main()