import json
import stats as stats_data
import copy
import re

input_file_path_raw = stats_data.raw_file_path

input_file_path = "data/final_results/clust_sum14_filtered-extra.json"
output_file_path = "data/final_results/clust_sum14_with_references.json"
output_file_path_filt = "data/final_results/clust_sum14_with_references_filtered.json"

def find_words_with_accent(sentence):
    words_with_accent = re.findall(r'\b\w*[́]\w*\b', sentence)

    words_with_accent_pro = combine_stressed_words(words_with_accent)
    return words_with_accent_pro

def combine_stressed_words(words):
  combined_words = []
  for i, word in enumerate(words):
    if i == 0:
      combined_words.append(word)
    else:
      if word[0] == "́":
        combined_words[-1] += word[1:]
      else:
        combined_words.append(word)
  return combined_words

          
def remove_accents_from_word(word_with_accent):
    word_without_accent = re.sub(r'[́]', '', word_with_accent) 
    return word_without_accent

def remove_accents_from_words(words_with_accent):
    return [remove_accents_from_word(w) for w in words_with_accent]
        

def find_entity(data, tgt_lemma):
    for ob in data:
        lemma_lower = remove_accents_from_word(ob["lemma"].lower()) 
        if lemma_lower == tgt_lemma:
            return ob
    return None

def get_entity_glosses(entity):
    try:
        glosses = []
        for synset in entity["synsets"]:
            glosses.append(synset["gloss"][0])
        return glosses
    except Exception as e:
        print(f"An error occurred: {e} |\n{entity}")
        return []


def pad_near_referenced_lemma(ref_lemma, gloss, referenced_glosses):
    # Знайти індекс першого входження цільового слова
    start_index = gloss.index(ref_lemma)
    # Знайти індекс кінця цільового слова + 1
    end_index = start_index + len(ref_lemma)
    padded_gloss = f"{gloss[:end_index]} ({'; '.join(referenced_glosses)}) {gloss[end_index:]}"
    return padded_gloss


def find_numbers(text):
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    return numbers

def get_numbers_in_between(tgt_gloss, words_with_accent):
    idxs = []
    for word in words_with_accent:
        start_index = tgt_gloss.index(word)
        end_index = start_index + len(word) 
        idxs.extend([start_index, end_index])

    strs_in_between = []
    for i, idx in enumerate(idxs):
        if i%2==0:
            continue
        if i != len(idxs)-1:
            strs_in_between.append(tgt_gloss[idx: idxs[i+1]])
        else:
            strs_in_between.append(tgt_gloss[idx: ])

    words_glosses_nums = []
    for st in strs_in_between:
        nums = find_numbers(st)
        if len(nums) == 0:
            nums = None
        words_glosses_nums.append(nums)
    return words_glosses_nums

def find_references(data, data_raw):
    words_with_accent_counter = 0
    modified_lemmas_counter = 0
    modified_lemmas = []
    for ob in data:
        target_lemma = ob["lemma"]
        # if remove_accents_from_word(target_lemma) == "алфавіт":
        #     print("stop")
        modified_lemma = False
        for synset in ob["synsets"]:
            tgt_gloss = synset["gloss"]
            words_with_accent = find_words_with_accent(tgt_gloss)

            if len(words_with_accent) == 0:
                continue

            
            words_with_accent_counter+=1

            words_without_accent = remove_accents_from_words(words_with_accent)
            # words_numbers = get_numbers_in_between(tgt_gloss, words_with_accent)
            
            words_numbers = get_numbers_in_between(remove_accents_from_word(tgt_gloss),                    
                                                   words_without_accent)
            

            for word_nums, word_with_accent in zip(words_numbers, words_with_accent):
                word_without_accent = remove_accents_from_word(word_with_accent)
                referenced_entity = find_entity(data_raw, word_without_accent)
                if not referenced_entity:
                    continue

                referenced_glosses = get_entity_glosses(referenced_entity)

                if len(referenced_glosses) == 0:
                    continue
                
                if word_nums == None or len(referenced_glosses) == 1:
                    padded_tgt_gloss = pad_near_referenced_lemma(word_without_accent, remove_accents_from_word(synset["gloss"]), referenced_glosses)
                    synset["gloss"] = padded_tgt_gloss
                    
                else:
                    # баг датасету (вервечка)
                    flitered_referenced_glosses = []
                    for num in word_nums:
                        if len(referenced_glosses) <= int(num)-1:
                            continue
                        flitered_referenced_glosses.append(referenced_glosses[int(num)-1])
                    # flitered_referenced_glosses = [referenced_glosses[int(num)-1] for num in word_nums]

                    padded_tgt_gloss = pad_near_referenced_lemma(word_without_accent, remove_accents_from_word(synset["gloss"]), flitered_referenced_glosses)
                    synset["gloss"] = padded_tgt_gloss
                    
                if not modified_lemma:
                    modified_lemmas_counter+=1
                modified_lemma = True
        if modified_lemma:
            modified_lemmas.append(ob)
    print(f"words_with_accent_counter {words_with_accent_counter}")
    print(f"modified_lemmas_counter {modified_lemmas_counter}")
    print(f"len(modified_lemmas) {len(modified_lemmas)}")
    return data, modified_lemmas




def main():
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = [json.loads(line) for line in input_file]

    with open(input_file_path_raw, 'r', encoding='utf-8') as input_file:
        data_raw = [json.loads(line) for line in input_file]

    output_data, mod_out_data = find_references(data, data_raw)

    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     for entry in output_data:
    #         json.dump(entry, output_file, ensure_ascii=False)
    #         output_file.write('\n')

    # with open(output_file_path_filt, 'w', encoding='utf-8') as output_file:
    #     json.dump({"referenced_lemmas_list": mod_out_data}, output_file, ensure_ascii=False)



if __name__ == '__main__':
    main()