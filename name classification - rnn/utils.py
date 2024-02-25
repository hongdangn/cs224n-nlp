import os
import unicodedata
import torch
import random
import string

ALL_LETTERS = string.ascii_letters + ".,;'-"
path = 'data/names'

def unicodeToASCII(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) \
                   if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, len(ALL_LETTERS))
    tensor[0][ALL_LETTERS.find(letter)] = 1
    return tensor

def line_to_tensor(line):
    line_tensor = torch.zeros(len(line), 1, len(ALL_LETTERS))
    for id in range(len(line)):
        line_tensor[id] = letter_to_tensor(line[id])   
    return line_tensor

def load_data(path):
    all_category = {filename: torch.tensor([i], dtype = torch.long) \
                    for i, filename in enumerate(os.listdir(path))}
    all_names, all_targets = [], []

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding = "utf-8") as file:
            content = file.read().splitlines()
        for line in content:
            all_names.append(line_to_tensor(line))
            all_targets.append(all_category[filename])

    return all_category, all_names, all_targets

if __name__ == '__main__':
    all_category, all_names, all_targets = load_data(path)
    print(all_names[:2], all_targets[:2])