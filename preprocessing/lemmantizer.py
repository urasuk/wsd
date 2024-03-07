import json
from collections import defaultdict
import stats as stats_data
import ast
import re

input_file_path = stats_data.lcopf_forms_found_file_path

# bad
# pairs_file_path = stats_data.pairs_file_path

# good
train_forms_file_path = stats_data.train_forms_file_path
test_forms_file_path = stats_data.test_forms_file_path


train_good_file_path = stats_data.train_lem_file_path
train_bad_file_path = stats_data.train_not_lem_file_path 

test_good_file_path = stats_data.test_lem_file_path
test_bad_file_path = stats_data.test_not_lem_file_path 


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
    
    def remove_punctuation(self, text, lemma):
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

    def get_target_idx(self, sentence_input, lemma):
        sentence = sentence_input.lower()
        # Tokenize the sentence into words
        words = [self.remove_punctuation(word, lemma) for word in sentence.split()]

        target_start_idx = -1
        target_end_idx = -1
        # target_start_pos = -1
        # target_end_pos = -1

        if lemma == 'замірити':
            print("bug")

        # ашгабатський абатський 🤒
        counter = 0
        for i, word in enumerate(words):
            # if word.startswith(lemma):
            if self.is_target_in_lemma_forms(word, lemma) or lemma in word: 
                #? or lemma in word:  автократ: правителя-автократа правителяавтократа
                # target_start_idx = i
                # target_end_idx = i + len(word.split("-"))  # Consider hyphenated words
                target_start_idx = sentence.find(word) # , target_start_pos + 1
                target_end_idx = target_start_idx + len(word) - 1

                counter+=1
                if counter > 2:
                    print(f'🤢 {sentence} | {lemma}')
                # ! випадок коли в речені декілька лем , треба брати мабуть усіх їх позиції щоб модель всі враховувала
                # 🤢 дзвоном скликався народ на віче. дзвоном вказували дорогу подорожнім, що заблукали в негоді. дзвоном оповіщався народ про перемогу і віталося переможне повернення полків з війни | дзвін
                
        if counter == 0: return False

        return {
            "target_start_idx": target_start_idx,
            "target_end_idx": target_end_idx,
            # "target_start_pos": target_start_pos,
            # "target_end_pos": target_end_pos
        }


def df_lemmantization(pairs_df, lemmatizer):
    good_data = []
    bad_data = []

    for i, pair in enumerate(pairs_df):
        sent1 = pair["sentence1"]
        sent2 = pair["sentence2"]
        lemma = pair["lemma"]

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

        if sent1_target_pos == False:
            # print(f'NO POSITIONS FOUND FOR SENT_1 | {sent1} | {lemma}')
            pass
            
        if sent2_target_pos == False:
            # print(f'NO POSITIONS FOUND FOR SENT_2 | {sent2} | {lemma}')
            pass
            
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

    #* Test: good [267_913] | bad [[161_25]]
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