import json

# Specify the file path
input_file_path = 'data/ua_gec.disamb.jsonl'
output_file_path = 'data/test_data.json'


READ_LINES_NUM = 10
# incorrect_matches_count = 0

def show_json_pretty(instance):
    pretty_json = json.dumps(instance, ensure_ascii=False, indent=2)
    print(pretty_json)


# Function to read and display the first 5 instances
def read_and_display_instances(file_path,  num_instances=READ_LINES_NUM):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read lines from the file
        lines = file.readlines()
        # Process each line (instance) in the file
        for line_num, line in enumerate(lines[:num_instances]):
            # Parse JSON from the line
            instance = json.loads(line.strip())

            show_json_pretty(instance)

            # parsed_dict = json.loads(instance)
            print(f"#{line_num+1} Example ----------------------------------------->")
            print(instance['sentence'], end='\n')

            for i, amb_inst in enumerate(instance['ambiguities']):
                print(f"Ambiguity ({i + 1}) word: '{amb_inst['word']}' ")
                for j, match_inst in enumerate(amb_inst['matches']):
                    print(f"\t\tMatch ({j+1}) ( Stresses: {match_inst['stress']} ) -> {match_inst['comment']}")

            for i, amb_inst in enumerate(instance['ambiguities']):
                isCorrectIndex = int(input(f'Choose correct match for ambiguity ({i + 1}) [* No correct? Write 0] : ')) - 1

                # if isCorrectIndex == -1:
                #     incorrect_matches_count += 1

                # instance['ambiguities'][i]['matches'][isCorrectIndex]['isCorrect'] = 1
                instance['ambiguities'][i]['correctMatchIndex'] = isCorrectIndex


            # Save the modified instance to a new file
            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                json.dump(instance, output_file, ensure_ascii=False)
                output_file.write('\n')  # Add a newline after each instance


if __name__ == "__main__":
    if input(f'Do you want to mark raw dataset ({READ_LINES_NUM} sentences) ? [yes/no] : ') == 'yes':
        # Clear the content of the output file before starting
        open(output_file_path, 'w').close()
        # Call the function to read, display, and save instances
        read_and_display_instances(input_file_path)
        # print(f'incorrect_matches_count: {incorrect_matches_count}')

