# ============================ RESULT OF EXECOTION get_stats.py ============================
# Statistics for the number of synsets per lemma:
# mean: 2.729896420456222
# std: 1.4351417262143809
# min: 2
# max: 24
# q1: 2.0
# q3: 3.0

# Statistics for the number of examples per lemma:
# mean: 3.090943485968781
# std: 2.42128175037582
# min: 1
# max: 33
# q1: 1.0
# q3: 4.0

# Total pairs number: 2216671 (було 2216636)     моя оцінка
# ============================ ++++++++++++++++++++++++++++++++ ============================


import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import stats as stats_data

input_file_path = stats_data.merged_file_path

def extract_info(data):
    num_synsets_list = [len(entry['synsets']) for entry in data]
    num_examples_list = [len(synset['examples']) for entry in data for synset in entry['synsets']]
    return {'synsets': num_synsets_list,
            "examples": num_examples_list }
    # return {"сенсів": num_synsets_list,
    #         "прикладів": num_examples_list }


# def plot_3d_histogram(data):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Осі X, Y, Z
#     x_data = list(data[:,0])  # Кількість сенсів
#     y_data = list(data[:,1])  # Кількість прикладів вживання
#     z_data = range(1, len(x_data) + 1)  # Кількість слів

#     # Побудова графіку
#     ax.bar3d(x_data, y_data, [0] * len(x_data), 1, 1, z_data, color='b', edgecolor='k')

#     # Настройки вісей
#     ax.set_xlabel('Кількість сенсів')
#     ax.set_ylabel('Кількість прикладів вживання')
#     ax.set_zlabel('Кількість слів')

#     plt.show()

import plotly.graph_objects as go

def plot_3d_histogram(data):
    # Розділення даних
    x_data = data[:, 0]  # Кількість сенсів
    y_data = data[:, 1]  # Кількість прикладів вживання
    z_data = list(range(1, 100))  # Кількість слів
    # z_data = list(range(1, len(x_data) + 1))  # Кількість слів

    # Побудова графіку
    fig = go.Figure(data=[go.Surface(z=z_data, x=x_data, y=y_data)])

    # Налаштування вісей
    fig.update_layout(scene=dict(
        xaxis_title='Кількість сенсів',
        yaxis_title='Кількість прикладів вживання',
        zaxis_title='Кількість слів'
    ))

    # Відображення графіку
    fig.show()

def extract_info_3d(data):
    final = []
    for entry in data:
        total_ex = 0
        for synset in entry['synsets']:
            total_ex += len(synset['examples'])
        final.append([len(entry['synsets']), total_ex])
    return np.array(final)

def calc_stats(num_items_list):
    return {
        "mean": np.mean(num_items_list),
        "std": np.std(num_items_list),
        "min": np.min(num_items_list),
        "max": np.max(num_items_list),
        "q1": np.percentile(num_items_list, 25),
        "q3": np.percentile(num_items_list, 75),
    }


def print_results(stats, label=''):
    print(f"Statistics for the number of {label} per lemma:")
    for stat_name, stat in stats.items():
        print(f"{stat_name}: {stat}")
    print("\n")


def plot(data, labelX='',labelY='', title=''):
    unique_values, counts = np.unique(data, return_counts=True)

    # 33884 for "synsets"    [ == len(data) ]  ===> Total number of entries
    # 92503 for "examples"   [ == len(data) ]  ===> Total number of senses
    # print(np.sum(counts)) 

    # 285920 for "examples" ===> Total number of examples
    # print(np.sum(data)) 

    # Adjust the width to add more space between bars
    width = 0.8
    
    plt.bar(unique_values, counts, width=width, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(labelX)
    # plt.ylabel("Number of lemmas")
    plt.ylabel(labelY)
    
    # Add labels above each bar
    for i, count in enumerate(counts):
        plt.text(unique_values[i], count + 0.1, str(count), ha='center', va='bottom')
    
    # Add x-values below each bar
    plt.xticks(unique_values, rotation=45, ha='right')
    
    plt.show()


def get_total_pairs_num(data):
    def _calculate_examples_per_lemma(data):
        num_examples_per_lemma = []

        for entry in data:
            lemma = entry['lemma']
            # Count the total number of examples for each lemma
            total_examples = sum(len(synset.get('examples')) for synset in entry.get('synsets'))

            num_examples_per_lemma.append(total_examples)

        return num_examples_per_lemma

    num_examples_per_lemma = _calculate_examples_per_lemma(data)
    unique_values, counts = np.unique(num_examples_per_lemma, return_counts=True)
    combinations_per_lemma = [comb(examples_quantity, 2, exact=True) for examples_quantity in unique_values]
    return np.dot(combinations_per_lemma, counts)


def main():
    # Read JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file] # len(data) == 33884

    info = extract_info(data)

    info3d = extract_info_3d(data)
    plot_3d_histogram(info3d)

    for label, num_item_list in info.items():
        stats = calc_stats(num_item_list)
        print_results(stats, label)
        # plot(num_item_list, label=f'Number of {label} per lemma', title=f'Histogram of {label} per lemma')
        label_text_X = "Кількість сенсів" if label == "synsets" else "Кількість прикладів"
        label_text_Y = "Кількість лем" if label == "synsets" else "Кількість сенсів"
        label_title = "Кількісний розподіл лем за кількістю сенсів" if label_title == "synsets" else "Кількісний розподіл сенсів за кількістю прикладів вживання"
        plot(num_item_list, labelX=label_text_X, labelY = label_text_Y, title=label_title) 
        # f'Гістограма к-сті {label_text} на {label_text2}'




    print(f"Total pairs number: {get_total_pairs_num(data)}")


if __name__ == '__main__':
    main()