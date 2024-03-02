import json
import stats as stats_data
import copy

input_file_path = stats_data.filtered_file_path
output_file_path = stats_data.filtered_extra_file_path

def get_synonims_template(lemma, acc_pos, toCopy=True):
    return {"lemma": lemma,
            "accent_positions": copy.deepcopy(acc_pos) if toCopy and isinstance(acc_pos, list) and len(acc_pos)!=0 else acc_pos}


def contains_dict(array_of_dicts, target_dict):
    for d in array_of_dicts:
        if d == target_dict:
            return True
    return False


def find_objects_with_equal_synsets(data):
    for ob in data:
        ob["synonims"] = [get_synonims_template(ob["lemma"], ob["accent_positions"])]

    for i, obj1 in enumerate(data):
        for j, obj2 in enumerate(data):
            if j <= i:
                continue
            # if i != j and \
            #    len(obj1["synsets"]) == len(obj2["synsets"]) and \
            #    obj1["synsets"] == obj2["synsets"] and \
            #    not contains_dict(obj1["synonims"],
            #                  get_synonims_template(obj2["lemma"], obj2["accent_positions"], False)):
            if len(obj1["synsets"]) == len(obj2["synsets"]) and obj1["synsets"] == obj2["synsets"]:
                obj1["synonims"].append(get_synonims_template(obj2["lemma"], obj2["accent_positions"]))
                obj2["synonims"].append(get_synonims_template(obj1["lemma"], obj1["accent_positions"]))
    return data


# def equal(path1, path2):
#     print("are they same: ")
#     with open(path1, 'r', encoding='utf-8') as input_file:
#         data1 = [json.loads(line) for line in input_file]
#     with open(path2, 'r', encoding='utf-8') as input_file:
#         data2 = [json.loads(line) for line in input_file]
#     for i, data_item1 in enumerate(data1):
#         if data_item1 != data2[i]:
#             print("bad")
#             return -1
#     print("yes")

def main():
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = [json.loads(line) for line in input_file]

    output_data = find_objects_with_equal_synsets(data)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entry in output_data:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    # equal("data/final_results/sum14_filtered-extra.json", "data/final_results/sum14_filtered-extra2.json")

if __name__ == '__main__':
    main()