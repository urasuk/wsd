import json
import random


def accuracy(pred, target):
    # Ensure the lengths of both lists are the same
    if len(pred) != len(target):
        raise ValueError("Lengths of 'pred' and 'target' must be the same.")

    # Calculate accuracy
    correct_predictions = sum(1 for p, t in zip(pred, target) if p == t)
    total_samples = len(pred)

    if total_samples == 0:
        raise ValueError("The length of 'pred' and 'target' should be greater than 0.")

    accuracy_value = correct_predictions / total_samples
    return accuracy_value


def model_random_sense(file_path):
    predictions = []
    correct_senses = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line_num, line in enumerate(lines):
            instance = json.loads(line.strip())
            for i, amb_inst in enumerate(instance['ambiguities']):
                predictions.append(random.randrange(0, len(amb_inst['matches'])))
                correct_senses.append(amb_inst['correctMatchIndex'])
    return predictions, correct_senses


if __name__ == "__main__":
    output_file_path = 'data/test_data.json'
    predictions, correct_senses = model_random_sense(output_file_path)
    for y_pred, y_true in zip(predictions, correct_senses):
        print(f'Predicted: {y_pred} | True: {y_true}')
    acc = accuracy(predictions, correct_senses)

    print(f'Accuracy [random]: {acc} ')
