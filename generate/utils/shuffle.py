import random

import regex as re


if __name__ == '__main__':
    data = open('data/mixed_input_raw.txt',
                encoding='utf-8', newline='\n').read()
    print(f'Loaded {len(data)} characters')
    chars = re.findall(r'\X|(?!\X)', data, re.U)
    print(f'Found {len(chars)} characters')

    random.shuffle(chars)

    with open('data/mixed_input.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(chars)
