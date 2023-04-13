import random
import string
import collections
import math
import matplotlib.pyplot as plt


def write(sequence):
    uniformity = ''
    source_excess = 0
    sequence_size = len(sequence)
    alphabet_size = len(set(sequence))
    counts = collections.Counter(sequence)
    probability = {symbol: count / sequence_size for symbol, count in counts.items()}
    mean_probability = sum(probability.values()) / len(probability)
    equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in probability.values())
    if equal:
        uniformity = 'рівна'
    else:
        uniformity = 'нерівна'
    entropy = -sum(p * math.log2(p) for p in probability.values())
    if alphabet_size > 1:
        source_excess = 1 - entropy / math.log2(alphabet_size)
    else:
        source_excess = 1
    str_probability = ', '.join([f"{symbol}={prob:.4f}" for symbol, prob in probability.items()])
    output = f'''Послідовність: {sequence}
Розмір послідовності: {sequence_size} bytes
Розмір алфавіту: {alphabet_size}
Ймовірності появи символів: {str_probability}
Середнє арифметичне ймовірностей: {round(mean_probability, 3)}
Ймовірність розподілу символів: {uniformity}
Ентропія: {round(entropy, 3)}
Надмірність джерела: {round(source_excess, 3)}
'''
    return alphabet_size, entropy, source_excess, uniformity, output


def unpacker_values(values, alphabet_sizes, entropies, sources_excess, uniformities, outputs):
    alphabet_sizes.append(values[0])
    entropies.append(values[1])
    sources_excess.append(values[2])
    uniformities.append(values[3])
    outputs.append(values[4])


alphabet_sizes = []
entropies = []
sources_excess = []
uniformities = []
outputs = []
sequences = []

N1 = 17
N_sequence = 100
N0 = N_sequence - N1
list1 = ['1'] * N1
list0 = ['0'] * N0
original_sequence_1 = list1 + list0
random.shuffle(original_sequence_1)
original_sequence_1 = ''.join(original_sequence_1)
sequences.append(original_sequence_1)
unpacker_values(write(original_sequence_1), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

list1 = list('яницький')
list0 = (N_sequence - len(list1)) * ['0']
original_sequence_2 = ''.join(list1 + list0)
sequences.append(original_sequence_2)
unpacker_values(write(original_sequence_2), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

list1 = list('яницький')
list0 = (N_sequence - len(list1)) * ['0']
original_sequence_3 = list1 + list0
random.shuffle(original_sequence_3)
original_sequence_3 = ''.join(original_sequence_3)
sequences.append(original_sequence_3)
unpacker_values(write(original_sequence_3), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

letters = list('яницький519')
n_letters = len(letters)
n_repeats = N_sequence // n_letters
remainders = N_sequence % n_letters
original_sequence_4 = ''.join((letters * n_repeats) + letters[0: remainders])
sequences.append(original_sequence_4)
unpacker_values(write(original_sequence_4), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

Pi = 0.2
letters = list('я' * int(N_sequence * Pi) + 'н' * int(N_sequence * Pi) + '5' * int(N_sequence * Pi) +
               '1' * int(N_sequence * Pi) + '9' * int(N_sequence * Pi))
random.shuffle(letters)
original_sequence_5 = ''.join(letters)
sequences.append(original_sequence_5)
unpacker_values(write(original_sequence_5), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

letters = ['я', 'н']
digits = ['5', '1', '9']
P_digit = 0.3
P_letters = 0.7
n_letters = round(N_sequence * P_letters)
n_digit = round(N_sequence * P_digit)
list_100 = [random.choice(letters) for i in range(0, n_letters)] + [random.choice(digits) for j in range(0, n_digit)]
original_sequence_6 = ''.join(list_100)
sequences.append(original_sequence_6)
unpacker_values(write(original_sequence_6), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

elements = string.ascii_lowercase + string.digits
list_100 = [random.choice(elements) for _ in range(0, N_sequence)]
original_sequence_7 = ''.join(list_100)
sequences.append(original_sequence_7)
unpacker_values(write(original_sequence_7), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

original_sequence_8 = ''.join(['1'] * N_sequence)
sequences.append(original_sequence_8)
unpacker_values(write(original_sequence_8), alphabet_sizes, entropies, sources_excess, uniformities,outputs)

results = []
for i in range(0, len(entropies)):
    results.append([alphabet_sizes[i], round(entropies[i], 2), round(sources_excess[i], 2), uniformities[i]])
N = len(results)
fig, ax = plt.subplots(figsize=(14/1.54, N/1.54))
headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
row = [f'Послідовність {i}' for i in range(1, N + 1)]
ax.axis('off')
table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')
table.set_fontsize(14)
table.scale(0.8, 2)
plt.savefig('Таблиця' + '.png', dpi=600)
plt.show()
with open('results_sequence.txt', 'w') as f:
    for i in outputs:
        f.writelines(i)
with open('sequences.txt', 'w') as f:
    for i in sequences:
        f.write(f'{i}\n')
