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


# print(remove_acute_accents("а́"))