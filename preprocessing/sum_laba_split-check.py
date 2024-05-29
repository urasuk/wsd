import stats as stats_data
import json
import matplotlib.pyplot as plt


# train_file_path = stats_data.train_file_path
# test_file_path = stats_data.test_file_path

# train_file_path = stats_data.train_lem_file_path
# test_file_path = stats_data.test_lem_file_path
# train_file_path = stats_data.train_forms_deduplicated_file_path
# test_file_path = stats_data.test_forms_deduplicated_file_path

train_file_path = stats_data.train_lem__dedup_file_path
test_file_path = stats_data.test_lem__dedup_file_path



def plot_pairs_count(sorted_pairs_counted, df_name):
    lemmas = [lemma for lemma, _ in sorted_pairs_counted]
    positive_counts = [counts["positive_pairs_n"] for _, counts in sorted_pairs_counted]
    negative_counts = [counts["negative_pairs_n"] for _, counts in sorted_pairs_counted]

    # Set up positions for curves
    positions = range(len(lemmas))

    # Plotting the curves
    # plt.plot(positions, positive_counts, label='Positive Pairs', linestyle='-', color='g')
    # plt.plot(positions, negative_counts, label='Negative Pairs', linestyle='-', color='r', alpha=0.7)

    # plt.plot(positions, positive_counts, label='Позитивні пари', linestyle='-', color='g')
    plt.plot(positions, positive_counts, label='Позитивні пари', linestyle='-', color='g', linewidth=2)
    plt.plot(positions, negative_counts, label='Негативні пари', linestyle='-', color='r', alpha=0.7, linewidth=0.5)

    # plt.scatter(positions, positive_counts, label='Позитивні пари', color='g', marker='o')
    # plt.scatter(positions, negative_counts, label='Негативні пари', color='r', marker='x', alpha=0.7)


    # Adding labels and title
    plt.xlabel('Леми')
    plt.ylabel('К-сть пар')
    # plt.title(f'[{df_name}] Positive and Negative Pairs Count for Each Lemma')
    plt.title(f'[{df_name}] К-сть позитивних та негативних пар для Кожної леми')


    # Hide x-axis tick labels
    plt.xticks([])

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def plot_pairs_count_sub(sorted_pairs_counted, df_name):
    lemmas_range = [0,10]
    sorted_pairs_counted_show = sorted_pairs_counted[lemmas_range[0]:lemmas_range[1]]
    lemmas = [lemma for lemma, _ in sorted_pairs_counted_show]
    positive_counts = [counts["positive_pairs_n"] for _, counts in sorted_pairs_counted_show]
    negative_counts = [counts["negative_pairs_n"] for _, counts in sorted_pairs_counted_show]

    bar_width = 0.35
    index = range(len(lemmas))

    fig, ax = plt.subplots()
    # positive_bars = ax.bar(index, positive_counts, bar_width, label='Positive Pairs')
    # negative_bars = ax.bar([i + bar_width for i in index], negative_counts, bar_width, label='Negative Pairs')
    positive_bars = ax.bar(index, positive_counts, bar_width, label='Позитивні пари')
    negative_bars = ax.bar([i + bar_width for i in index], negative_counts, bar_width, label='Негативні пари')


    ax.set_xlabel('Lemma')
    ax.set_ylabel('Count')
    # ax.set_title(f'[{df_name}] Counts of Positive and Negative Pairs for lemmas in range [{lemmas_range[0]}; {lemmas_range[1]}]')
    ax.set_title(f'[{df_name}] К-сть позитивних та негативних пар для лем в проміжку: [{lemmas_range[0]}; {lemmas_range[1]}]')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(lemmas)
    ax.legend()
    plt.show()

def count_pos_neg_pairs(pairs, df_name):
    pairs_counted = {}

    for pair in pairs:
        lemma = pair["lemma"]

        # Initialize the lemma if it's not in pairs_counted
        if lemma not in pairs_counted:
            pairs_counted[lemma] = {"positive_pairs_n": 0, "negative_pairs_n": 0}

        # Count positive and negative pairs
        if pair["label"] == 1:
            pairs_counted[lemma]["positive_pairs_n"] += 1
        elif pair["label"] == 0:
            pairs_counted[lemma]["negative_pairs_n"] += 1

    sorted_pairs_counted = sorted(pairs_counted.items(), key=lambda x: (x[1]["positive_pairs_n"],
                                                                        x[1]["negative_pairs_n"]),
                                                                        reverse=True)

    total_pos_n = 0
    total_neg_n = 0
    for entry in sorted_pairs_counted:
        total_pos_n+=entry[1]["positive_pairs_n"]
        total_neg_n+=entry[1]["negative_pairs_n"]
    print(f"[{df_name}] Total number of Positive [✚] pairs: {total_pos_n}")  # 570_077
    print(f"[{df_name}] Total number of Negative [-] pairs: {total_neg_n}")  # 1_646_559

    return sorted_pairs_counted, total_pos_n, total_neg_n


def main():
    with open(train_file_path, 'r', encoding='utf-8') as file:
        train_pairs = [json.loads(line) for line in file]
    with open(test_file_path, 'r', encoding='utf-8') as file:
        test_pairs = [json.loads(line) for line in file]

    # для трен та тест наборів:
    # для кожної леми порахувати к-сть позитивних та негативних прикладів
    # знайти сумарну к-сть + та - прикладів для кожного набору
    # посортувати пари по к-сті + та - в спад порядку і вивести для кожної леми 2 стовбчики (для + та -), де по y - ксть

    df_name = 'Train'
    df_name = 'Тренувальний набір'
    sorted_pairs_counted, total_pos_n, total_neg_n = count_pos_neg_pairs(train_pairs, df_name)
    plot_pairs_count(sorted_pairs_counted, df_name)
    plot_pairs_count_sub(sorted_pairs_counted, df_name)

    df_name = 'Тестувальний набір'
    sorted_pairs_counted, total_pos_n, total_neg_n = count_pos_neg_pairs(test_pairs, df_name)
    plot_pairs_count(sorted_pairs_counted, df_name)
    plot_pairs_count_sub(sorted_pairs_counted, df_name)


if __name__ == '__main__':
    main()

# [Тренувальний набір] Total number of Positive [✚] pairs: 318084
# [Тренувальний набір] Total number of Negative [-] pairs: 320028
# [Тестувальний набір] Total number of Positive [✚] pairs: 35200
# [Тестувальний набір] Total number of Negative [-] pairs: 35357