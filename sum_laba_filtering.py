import json
import stats as stats_data

input_file_path = stats_data.raw_file_path
output_file_path = stats_data.filtered_file_path

MIN_LEMMA_LEN = stats_data.MIN_LEMMA_LEN
MIN_GLOSS_LEN = stats_data.MIN_GLOSS_LEN
MIN_EXAMPLE_LEN = stats_data.MIN_EXAMPLE_LEN
MIN_EXAMPLE_NUM = stats_data.MIN_EXAMPLE_NUM
MIN_SYNSET_NUM = stats_data.MIN_SYNSET_NUM
MIN_SYNSET_NUM_PRIME = stats_data.MIN_SYNSET_NUM_PRIME


def show_json_pretty(instance):
    pretty_json = json.dumps(instance, ensure_ascii=False, indent=2)
    print(pretty_json)


def find_acute_accent_positions(input_string):
    positions = []

    for idx, char in enumerate(input_string):
        if char == '́':
            positions.append(idx-1) # cuz idx - is index of acute symbol
    return positions


def remove_acute_accents(input_string, accent_positions=None):
    accent_positions = accent_positions if accent_positions else find_acute_accent_positions(input_string)

    for i, position in enumerate(accent_positions):
        position -= i
        input_string = input_string[:position + 1] + input_string[position + 2:]

    return input_string


def remove_non_ukrainian_symbols(input_string):
    # allowed_symbols = set("́-")  # Acute accent and hyphen are allowed
    allowed_symbols = set("-")  # Acute accent and hyphen are allowed

    ukrainian_letters = set("АБВГҐДЕЄЖЗИIІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиiіїйклмнопрстуфхцчшщьюя")

    result_string = ''.join(char for char in input_string if char in ukrainian_letters or char in allowed_symbols)

    return result_string


def gloss_handling(gloss_arr):
    # --- GLOSS HANDLING --->
    glosses_num = len(gloss_arr)
    if glosses_num == 0:
        return False

    gloss_joined = ' '.join(gloss_arr)
    gloss_joined = remove_acute_accents(gloss_joined)  # some glosses can have words with accents
    if len(gloss_joined) < glosses_num - 1 + MIN_GLOSS_LEN:
        return False
    return gloss_joined


def examples_handling(examples_arr, new_synset):
    # --- EXAMPLES HANDLING --->
    examples_num = len(examples_arr)
    if examples_num < MIN_EXAMPLE_NUM:
        return False

    new_synset["examples"] = []

    for ex in examples_arr:
        if len(ex["ex_text"]) < MIN_EXAMPLE_LEN:
            return False
        ex["ex_text"] = remove_acute_accents(ex["ex_text"])  # some examples MAYBE have words with accents
        new_synset["examples"].append(ex)

    if len(new_synset["examples"]) < MIN_EXAMPLE_NUM:
        return False

    return True


def filter_entry(entry):

    accent_positions = find_acute_accent_positions(entry['lemma'])
    lemma_acute_removed = remove_acute_accents(entry['lemma'], accent_positions)
    synsets = entry.get('synsets', [])

    lemma_acute_removed = remove_non_ukrainian_symbols(lemma_acute_removed)

    if len(lemma_acute_removed) <= MIN_LEMMA_LEN:
        return None


    filtered_entry = {
        "lemma": lemma_acute_removed.lower(),  # ⛔️ toLowerCase
        "accent_positions": accent_positions if len(accent_positions) != 0 else None,
        "synsets": []
    }

    has_prime = int(entry["prime"]) if entry["prime"] else False  # 🔥🔥🔥

    if len(synsets) < MIN_SYNSET_NUM and not has_prime:
        return None

    # --- SYNSETS HANDLING --->
    for synset in synsets:
        new_synset = {}
        # --- GLOSS HANDLING --->
        gloss_joined = gloss_handling(synset["gloss"])
        if not gloss_joined:
            continue
        new_synset["gloss"] = gloss_joined

        # --- EXAMPLES HANDLING --->
        if not examples_handling(synset["examples"], new_synset):
            continue

        filtered_entry["synsets"].append(new_synset)

    # if len(filtered_entry["synsets"]) < MIN_SYNSET_NUM and not has_prime:
    if (has_prime and len(filtered_entry["synsets"]) < MIN_SYNSET_NUM_PRIME) or\
       (not has_prime and len(filtered_entry["synsets"]) < MIN_SYNSET_NUM):
        return None
    return filtered_entry


def main():
    num_objects_input = 0
    num_objects_output = 0
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:

        for i, line in enumerate(input_file):
            num_objects_input = i

            entry = json.loads(line)  # Parse the JSON object from the line
            # if entry["lemma"] == "А":  # АБА́З ПО́ТЯГ
            #     print("❤️‍  🔥")
            filtered_entry = filter_entry(entry)
            if filtered_entry:
                json.dump(filtered_entry, output_file, ensure_ascii=False)
                output_file.write('\n')
                num_objects_output += 1

        print(f'Number of objects in INPUT file: {num_objects_input}')
        print(f'Number of objects in FILTERED file: {num_objects_output}')


if __name__ == '__main__':
    main()