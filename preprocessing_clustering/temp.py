import re

def extract_word_and_number(text):
    # Шукаємо слово та число у тексті
    match = re.search(r'(\w+)\s+(\d+)(?:,\s*(\d+))?', text)
    # match2 = re.search(r'\b\d+(?:,\s*\d+)*\b', text)
    numbers = re.findall(r'\b\d+(?:,\s*\d+)*\b', text)


    if match:
        # Якщо знайдено одне число
        if match.group(3) is None:
            word = match.group(1)
            number = int(match.group(2))
            return (word, number)
        # Якщо знайдено кілька чисел
        else:
            word = match.group(1)
            numbers = [int(match.group(2))] + [int(match.group(3))]
            return (word, numbers)
    else:
        return None

# Приклади використання
# print(extract_word_and_number("Те саме, що підхожий 1"))  # ("підхожий", 1)
print(extract_word_and_number("Те 3 саме, що підхожий 1"))  # ("підхожий", [1, 2, 3])
