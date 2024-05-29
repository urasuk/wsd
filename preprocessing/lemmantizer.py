import json
from collections import defaultdict
import stats as stats_data
import ast
import re

input_file_path = stats_data.lcopf_forms_found_file_path

# bad
# pairs_file_path = stats_data.pairs_file_path

# good
# train_forms_file_path = stats_data.train_forms_file_path
# test_forms_file_path = stats_data.test_forms_file_path
train_forms_file_path = stats_data.train_forms_deduplicated_file_path
test_forms_file_path = stats_data.test_forms_deduplicated_file_path


# train_good_file_path = stats_data.train_lem_file_path
# train_bad_file_path = stats_data.train_not_lem_file_path 

# test_good_file_path = stats_data.test_lem_file_path
# test_bad_file_path = stats_data.test_not_lem_file_path 

train_good_file_path = stats_data.train_lem__dedup_file_path
train_bad_file_path = stats_data.train_not_lem__dedup_file_path 

test_good_file_path = stats_data.test_lem__dedup_file_path
test_bad_file_path = stats_data.test_not_lem__dedup_file_path 


class Lemmatizer:
    def __init__(self, jsonl_file_path):
        self.lemmatizer_map = {}
        self.load_from_jsonl(jsonl_file_path)

    def load_from_jsonl(self, jsonl_file_path):
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                lemma = entry.get('lemma')
                forms = entry.get('forms')
                if lemma and forms:
                    if type(forms) != list:
                        forms = ast.literal_eval(forms)
                    self.lemmatizer_map[lemma] = forms

    def is_target_in_lemma_forms(self, target, lemma):
        try:
            forms = self.lemmatizer_map.get(lemma, [])
            return target in forms 
        except Exception as e:
            print(f"🥶 ПРОБЛЕМА ЩО ЛЕМИ МАЮТЬ В synonyms такі леми, яких нема в merged_df: {e}") # наприклад ЗАМІ́РИТИ
            return False
    
    def remove_punctuation(self, text):
        # Define the punctuation characters to remove
        punctuation_to_remove = set(";:,.!\"?/“”()«»") # -

        # Add "-" to punctuation_to_remove if lemma contains "-"
        # if '-' in lemma:
        #     punctuation_to_remove.remove('-')

        result = []

        for char in text:
            if char not in punctuation_to_remove: # or (char == '-' and '-' in result)
                result.append(char)

        return ''.join(result)

    def remove_extra_spaces(self, sentence):
        # Використовуємо регулярний вираз для заміни двох або більше пробілів підряд на один пробіл
        return re.sub(r'\s{2,}', ' ', sentence).strip()

    def find_targets_general_position(self, sentence):
        words = sentence.split()
        indices = []
        start = 0
        for w in words:
            end = start + len(w) - 1
            if len(w) >= 3:  # Фільтрація за довжиною слова
                indices.append((w, start, end))
            start = end + 2  # додаємо 1 за словом і 1 за пробілом
        return indices

    def find_targets_specific_position(self, targets_with_indeces):
        result_data = []
        for target, idx_start_gen, idx_end_gen in targets_with_indeces:
            target_cleaned = self.remove_punctuation(target)
            idx_start_sub = target.find(target_cleaned)
            idx_end_sub = idx_start_sub + len(target_cleaned) - 1
            result_data.append((target_cleaned,
                               idx_start_gen+idx_start_sub,
                               idx_start_gen+idx_end_sub))
        return result_data


    def get_target_idx(self, sentence_input, lemma):
        # sentence = sentence_input.lower() #? 📍
        sentence = sentence_input

        targets_with_idxs_gen = self.find_targets_general_position(sentence)
        targets_with_idxs = self.find_targets_specific_position(targets_with_idxs_gen)

        # Tokenize the sentence into words
        # targets = [self.remove_punctuation(word) for word in sentence.split()]

        target_start_idx = -1
        target_end_idx = -1

        # if lemma == 'замірити':
        #     print("bug")

        # ашгабатський абатський 🤒
        counter = 0
        for i, (target, s_idx, e_idx) in enumerate(targets_with_idxs):
            target_lower = target.lower() #? 📍
            if self.is_target_in_lemma_forms(target_lower, lemma) \
               or (lemma in target_lower and target_lower.find('-') != -1 and lemma.find('-') != -1): 
               #? пробую новий підхід
                
                # or lemma in word:
                #? or lemma in word:  автократ: правителя-автократа правителяавтократа
                #! ⭐️ призвело до наступної проблеми:
                # sent1 Провести судно через скелясті вузькі ✅ворота✅ з підводними рифами і мілинами вимагало від капітана великої майстерності та досвіду
                # sent2 Хлопці грали у футбола  . ⛔️Ворота⛔️рем був Василь... М'яч летів просто на ✅ворота✅


                # target_start_idx = i
                # target_end_idx = i + len(target.split("-"))  # Consider hyphenated words

                #! ⭐️ another problem
                # target_start_idx = sentence.find(target) # , target_start_pos + 1
                # target_end_idx = target_start_idx + len(target) - 1
                target_start_idx = s_idx
                target_end_idx = e_idx

                counter+=1
                if counter > 2:
                    print(f'🤢 {sentence} | {lemma}')
                    break
                # ! випадок коли в речені декілька лем , треба брати мабуть усіх їх позиції щоб модель всі враховувала
                # 🤢 дзвоном скликався народ на віче. дзвоном вказували дорогу подорожнім, що заблукали в негоді. дзвоном оповіщався народ про перемогу і віталося переможне повернення полків з війни | дзвін
                
        if counter == 0: return False

        return {
            "target_start_idx": target_start_idx,
            "target_end_idx": target_end_idx,
        }


def df_lemmantization(pairs_df, lemmatizer):
    good_data = []
    bad_data = []

    for i, pair in enumerate(pairs_df):
        pair["sentence1"] = lemmatizer.remove_extra_spaces(pair["sentence1"])
        pair["sentence2"] = lemmatizer.remove_extra_spaces(pair["sentence2"])
        sent1 = pair["sentence1"]
        sent2 = pair["sentence2"]
        # lemma = pair["lemma"]

        # if lemma == 'автократ' and \
        #    sent1 == 'Перед своїм вторгненням до Греції східний автократ наслухався порад, як позбавити еллінів продовольчої бази, і тверезо переслідував цю стратегічну мету' and \
        #    sent2 == 'Більша частина державних ресурсів стала особистою власністю правителя-автократа і його найближчого оточення':
        #     print("#")
        synonyms = pair["synonyms"]
        sent1_target_pos = False
        sent2_target_pos = False
        

        for synonym in synonyms:
            if not sent1_target_pos:
                sent1_target_pos = lemmatizer.get_target_idx(sent1, synonym["lemma"])
            if not sent2_target_pos:
                sent2_target_pos = lemmatizer.get_target_idx(sent2, synonym["lemma"])

        # if sent1_target_pos == False:
        #     print(f'NO POSITIONS FOUND FOR SENT_1 | {sent1} | {lemma}')
            
        # if sent2_target_pos == False:
        #     print(f'NO POSITIONS FOUND FOR SENT_2 | {sent2} | {lemma}')
            
        if sent1_target_pos == False or sent2_target_pos == False:
            bad_data.append(pair)
            continue
        
        # If both positions are found, add to good_data
        pair["sent1_target_pos"] = sent1_target_pos
        pair["sent2_target_pos"] = sent2_target_pos
        good_data.append(pair)

    return good_data, bad_data
        


def main():
    lemmatizer = Lemmatizer(input_file_path)

    # Lemmatize a word
    # res = lemmatizer.get_target_idx(".,лікарка-тому? потягами,.", "потяг")
    # res = lemmatizer.get_target_idx("Хлопці грали у футбола . Воротарем був Василь... М'яч летів просто на ворота",
    #                                 "ворота")
    
    # res = lemmatizer.get_target_idx("Од дороги його відокремлювала неглибока стара канава, поросла ліщиною і лозами", 
                                    # "відокремлювати")
    # print(res)

    # =============================================================================

    with open(test_forms_file_path, 'r', encoding='utf-8') as file:
        pairs_test_df = [json.loads(line) for line in file]

    test_good, test_bad = df_lemmantization(pairs_test_df, lemmatizer)

    # Write the test set to a new file
    with open(test_good_file_path, 'w', encoding='utf-8') as output_file:
        for entry in test_good:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    # Write the test set to a new file
    with open(test_bad_file_path, 'w', encoding='utf-8') as output_file:
        for entry in test_bad:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    #* Test: good [267_913] | bad [[16_125]]
    #* after changing: Test: good [264_518] | bad [[19_520]]
    print(f"Test: good [{len(test_good)}] | bad [[{len(test_bad)}]]")
    # =============================================================================

    with open(train_forms_file_path, 'r', encoding='utf-8') as file:
        pairs_train_df = [json.loads(line) for line in file]

    train_good, train_bad = df_lemmantization(pairs_train_df, lemmatizer)

    # Write the train set to a new file
    with open(train_good_file_path, 'w', encoding='utf-8') as output_file:
        for entry in train_good:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    # Write the train set to a new file
    with open(train_bad_file_path, 'w', encoding='utf-8') as output_file:
        for entry in train_bad:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    #* Train: good [802_065] | bad [[50_045]]
    #* after changing: Train: good [797_629] | bad [[54_481]]
    print(f"Train: good [{len(train_good)}] | bad [[{len(train_bad)}]]")
    # =============================================================================



    # with open(pairs_file_path, 'r', encoding='utf-8') as file:
    #     pairs_data = [json.loads(line) for line in file]

    # for line in pairs_data:
    #     lemma, pairs = next(iter(line.items()))
    #     # line_expanded = {
    #     #     "lemma": lemma,
    #     #     "pairs": pairs,
    #     #     "positive_pairs": [],
    #     #     "negative_pairs": [],
    #     #     }
 
    #     for i, pair in enumerate(pairs):
    #         sent1 = pair["sentence1"]
    #         sent2 = pair["sentence2"]
    #         lemma = pair["lemma"]

    #         # if lemma == 'автократ' and \
    #         #    sent1 == 'Перед своїм вторгненням до Греції східний автократ наслухався порад, як позбавити еллінів продовольчої бази, і тверезо переслідував цю стратегічну мету' and \
    #         #    sent2 == 'Більша частина державних ресурсів стала особистою власністю правителя-автократа і його найближчого оточення':
    #         #     print("#")
    #         synonyms = pair["synonyms"]
    #         sent1_target_pos = False
    #         sent2_target_pos = False
    #         for synonym in synonyms:
    #             if not sent1_target_pos:
    #                 sent1_target_pos = lemmatizer.get_target_idx(sent1, synonym["lemma"])
    #             if not sent2_target_pos:
    #                 sent2_target_pos = lemmatizer.get_target_idx(sent2, synonym["lemma"])

    #         if sent1_target_pos == False:
    #             print(f'NO POSITIONS FOUND FOR SENT_1 | {sent1} | {lemma}')
    #         if sent2_target_pos == False:
    #             print(f'NO POSITIONS FOUND FOR SENT_2 | {sent2} | {lemma}')
            

if __name__ == '__main__':
    main()

##
# Tddest: good [70557] | bad [[4981]]
# Train: good [638112] | bad [[41720]]
# Total: 708,669
