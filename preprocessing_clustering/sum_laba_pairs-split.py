# 1) Підраховуємо к-сть позитивних та негативних пар для кожної леми
# 2) Підраховуємо загальну к-сть позитивних (X) та негативних (3X) пар
# 3) Формування тренувальних та тестувальних наборів (відношення 80/20) з урахуванням наступних факторів:
# ---> 1. Пари тренувальних та тестувальних наборів не повинні мати спільні леми, а саме:
# -------> 1.1. У тренувальному наборі використовуватимуться перші X лем відсортованих
#               за кількістю позитивних прикладів у спад. порядку, за умови, що сумарна к-сть позитивних пар цих X лем 
#               не перевищує 80% усіх доступних позитивних пар. Тобто використовуватимуться леми з 
#               найбільшою к-стю позитивних пар, що дозволить моделі навчатися на різноманітних позитивних зразках і 
#               покращить її здатність визначати схожість! для нових пар.
# ---> 2. К-сть позитивних та негативних пар у обох наборах повинна бути збалансована
# -------> 2.1. Оскільки к-сть негативних пар ~3 рази більша за к-сть позитивних пар, то процес формування train/test 
#               наборів даних повинен відштовхуватися від к-сті ПОЗИТИВНИХ пар,
#               а зайві негативні пари для кожної леми - відкидаються, для забезпечення балансу + та - пар.
# ---> 3. К-сть позитивних та негативних пар для кожної леми - наближується! до збалансованої


import json
import random
import stats as stats_data
import numpy as np
import math

import matplotlib.pyplot as plt


# input_file_path = stats_data.pairs_file_path
# train_file_path = stats_data.train_file_path
# test_file_path = stats_data.test_file_path

input_file_path = stats_data.pairs_forms_file_path
train_file_path = stats_data.train_forms_file_path
test_file_path = stats_data.test_forms_file_path

# Set the seed for reproducibility
random.seed(42)

def plot_pos_neg_quantity(pos_pairs_quantity, neg_pairs_quantity):
    if len(pos_pairs_quantity) != len(neg_pairs_quantity):
        raise ValueError("Both lists should have the same length.")
    
    plt.plot(pos_pairs_quantity, neg_pairs_quantity, marker='o', linestyle='-', color='r')
    plt.xlabel("Positive pairs quantity")
    plt.ylabel("Negative pairs quantity")
    plt.title("Each point on plot is Lemma a with some + and - usage examples")
    plt.grid(True)
    plt.show()

def count_pos_neg_pairs(dataset, get_plot=False):
    dataset_expanded = []

    for line in dataset:
        lemma, pairs = next(iter(line.items()))
        line_expanded = {
            "lemma": lemma,
            "pairs": pairs,
            "positive_pairs": [],
            "negative_pairs": [],
            }
 
        for i, pair in enumerate(pairs):
            if pair["label"] == 1:
                line_expanded["positive_pairs"].append(pair)
            elif pair["label"] == 0:
                line_expanded["negative_pairs"].append(pair)

        line_expanded["positive_n"] = len(line_expanded["positive_pairs"])
        line_expanded["negative_n"] = len(line_expanded["negative_pairs"])
        dataset_expanded.append(line_expanded)

    #% =============================================== NOTEWORTHY ==============================================
    #%
    #% Оскільки загалом к-сть позитивних пар майже в 3 менша за к-сть негативних пар
    #% Total number of Positive [✚] pairs: 570077
    #% Total number of Negative [-] pairs: 1646559
    #% то важливо щоб у тренувальних даних було достатньо позитивних пар 
    #% для ефективного тренування моделі.
    #% Вибір перших лем з відсортованого списку за кількістю позитивних прикладів: може бути відмінним підходом для забезпечення того, що у тренувальних даних є достатньо позитивних прикладів для кожної леми. Це дозволить моделі навчатися на різноманітних позитивних зразках і покращить її здатність визначати схожість для нових пар.
    #%
    dataset_sorted = sorted(dataset_expanded, key=lambda x: (x["positive_n"], x["negative_n"]), reverse=True)

    if get_plot:
        pos_pairs_quantity = []
        neg_pairs_quantity = []
        for entry in dataset_sorted:
            pos_pairs_quantity.append(entry["positive_n"])
            neg_pairs_quantity.append(entry["negative_n"])
        plot_pos_neg_quantity(pos_pairs_quantity, neg_pairs_quantity)

    total_pos_n = 0
    total_neg_n = 0
    for entry in dataset_sorted:
        total_pos_n+=entry["positive_n"]
        total_neg_n+=entry["negative_n"]
    print(f"Total number of Positive [✚] pairs: {total_pos_n}")  # 570_077
    print(f"Total number of Negative [-] pairs: {total_neg_n}")  # 1_646_559

    return dataset_sorted, total_pos_n, total_neg_n


def _filter_elements_by_indexes(elements, indexes):
    return [elements[i] for i in indexes]


def _calc_number_of_examples(lemmas):
    sum_pos = 0
    sum_neg = 0
    for lemma_ex in lemmas:
        sum_pos += lemma_ex["positive_n"]
        sum_neg += lemma_ex["negative_n"]
    return sum_pos, sum_neg


def _remove_random_elements(input_list, n):
    # ! 🌞 -> у трейні багато {pos_n 0 neg_n 1}
    #!   що робити з такими? лишати чи ні ?
    if n == 0:  #! лишати
        return input_list, []
    
    if n >= len(input_list):
        return input_list, []

    random.shuffle(input_list)

    return input_list[:n], input_list[n:]

def equalize_pos_neg_examples(lemmas_extended, positive_examples_n, df_type):
    lemmas_sorted = sorted(lemmas_extended, key=lambda x: (x["negative_n"], -x["positive_n"]))
  
    for lemma_ex in lemmas_sorted:
        positive_n = lemma_ex["positive_n"]
        lemma_ex["negative_pairs_new"], lemma_ex["negative_pairs_extra"] = _remove_random_elements(lemma_ex["negative_pairs"], positive_n)
        lemma_ex["negative_n_new"] = len(lemma_ex["negative_pairs_new"])

    sum_neg = 0
    for lemma_ex in lemmas_sorted:
        sum_neg += lemma_ex["negative_n_new"]
    # print(f"Train: New number of negative examples: {sum_neg}")

    if sum_neg < positive_examples_n:
        lemmas_extra = []

        for lemma_ex in lemmas_sorted:
            positive_n = lemma_ex["positive_n"]
            if lemma_ex["negative_n_new"] < lemma_ex["negative_n"]:
                lemmas_extra.append(lemma_ex)

     
        for i in range (positive_examples_n - sum_neg):
            rand_lemma = {"negative_pairs_extra": []}
            while not rand_lemma["negative_pairs_extra"]:
                rand_lemma = random.choice(lemmas_extra)
                
            new_ex = random.choice(rand_lemma["negative_pairs_extra"])
            rand_lemma["negative_pairs_new"].append(new_ex)
            rand_lemma["negative_n_new"] = len(rand_lemma["negative_pairs_new"])
            rand_lemma["negative_pairs_extra"].remove(new_ex)

    sum_neg = 0
    for lemma_ex in lemmas_sorted:
        sum_neg += lemma_ex["negative_n_new"]
    print(f"🔥 {df_type}: New number of negative examples: {sum_neg}")

    return lemmas_sorted


# ! у тесті не має бути тих лем що в трейн 
def split_jsonl(input_file, train_file, test_file, split_ratio=0.75):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = [json.loads(line) for line in file]
        
    lemmas_sorted, total_pos_n, total_neg_n = count_pos_neg_pairs(lines)  
    #% Процес формування train/test наборів даних повинен відштовхуватися від к-сті ПОЗИТИВНИХ пар,
    #% а оскільки к-сть негативних пар у нас достатньо, то ми можемо їх відкидати, для забезпечення балансу + та -
    #% У додаток в тренувальному наборі, як згадувалося вище, має бути леми з великою к-стю позитивних пар

    eval_train_pos_n = math.ceil(total_pos_n*split_ratio)
    eval_test_pos_n = total_pos_n - eval_train_pos_n

    train_lemmas_ids = []
    test_lemmas_ids = []
    quantity_i = 0
    for i, lemma_extended in enumerate(lemmas_sorted):
        if quantity_i+lemma_extended["positive_n"] < eval_train_pos_n: # ! через це 🌞 у трейні багато pos 0 neg 1
            train_lemmas_ids.append(i)
            quantity_i+=lemma_extended["positive_n"]
        else:
            test_lemmas_ids.append(i)
        
    train_lemmas_extened = _filter_elements_by_indexes(lemmas_sorted, train_lemmas_ids)
    test_lemmas_extened = _filter_elements_by_indexes(lemmas_sorted, test_lemmas_ids)

    actual_train_pos_n, actual_train_neg_n = _calc_number_of_examples(train_lemmas_extened)
    actual_test_pos_n, actual_test_neg_n = _calc_number_of_examples(test_lemmas_extened)
    print(f"Train actual: positive🔥 = {actual_train_pos_n}  (eval: {eval_train_pos_n}) | negative = {actual_train_neg_n}")
    print(f"Test  actual: positive🔥 = {actual_test_pos_n}  (eval: {eval_test_pos_n}) | negative = {actual_test_neg_n}")
    print(f"Total pairs: {actual_train_pos_n + actual_train_neg_n + actual_test_pos_n + actual_test_neg_n }")
    #% Train actual: positive = 427_557 (eval: 427_558) | negative = 1_384_267
    #% Test actual: positive  = 142_520 (eval: 142_519) | negative = 262_292
    #% Total pairs: 2216636 ✅


    #% Тепер у train/test наборах даних зрівнюємо к-сть негативних пар до к-сті позитивних
    train_lemmas_eq = equalize_pos_neg_examples(train_lemmas_extened, actual_train_pos_n, "Train")
    test_lemmas_eq = equalize_pos_neg_examples(test_lemmas_extened, actual_test_pos_n, "Test")
    
    train_final = []
    for lemma_entry in train_lemmas_eq:
        train_final.extend(lemma_entry["negative_pairs_new"] + lemma_entry["positive_pairs"])
    random.shuffle(train_final)

    test_final = []
    for lemma_entry in test_lemmas_eq:
        test_final.extend(lemma_entry["negative_pairs_new"] + lemma_entry["positive_pairs"])
    random.shuffle(test_final)


    print(f"Train final: {len(train_final)}") # 427557 * 2 = 855114
    print(f"Test final: {len(test_final)}")   # 142520 * 2 = 285040

    # Write the train set to a new file
    with open(train_file, 'w', encoding='utf-8') as train_output_file:
        # train_output_file.writelines(train_final)
        for entry in train_final:
            json.dump(entry, train_output_file, ensure_ascii=False)
            train_output_file.write('\n')

    # Write the test set to a new file
    with open(test_file, 'w', encoding='utf-8') as test_output_file:
        # test_output_file.writelines(test_final)
        for entry in test_final:
            json.dump(entry, test_output_file, ensure_ascii=False)
            test_output_file.write('\n')


def main():
    split_jsonl(input_file_path, train_file_path, test_file_path)


if __name__ == '__main__':
    main()


