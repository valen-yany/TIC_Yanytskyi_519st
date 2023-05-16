import collections
import math

from matplotlib import pyplot as plt


def float_bin(point, size_cod):
    binary_code = ''
    for x in range(size_cod):
        point = point * 2
        if point > 1:
            binary_code = binary_code + str(1)
            x = int(point)
            point = point - x
        if point < 1:
            binary_code = binary_code + str(0)
        if point == 1:
            binary_code = binary_code + str(1)
    return binary_code


def encode_ac(uniq_chars, probabilitys, alphabet_size, seq):
    alphabet = list(uniq_chars)
    prob = [probabilitys[symbol] for symbol in alphabet]
    unity = []
    probability_range = 0.0

    for i in range(alphabet_size):
        l = probability_range
        probability_range = probability_range + prob[i]
        u = probability_range
        unity.append([alphabet[i], l, u])
    for i in range(len(seq) - 1):
        for j in range(len(unity)):
            if seq[i] == unity[j][0]:
                probability_low = unity[j][1]
                probability_high = unity[j][2]
                diff = probability_high - probability_low
                for k in range(len(unity)):
                    unity[k][1] = probability_low
                    unity[k][2] = prob[k] * diff + probability_low
                    probability_low = unity[k][2]
                break
    low = 0
    high = 0
    for i in range(len(unity)):
        if unity[i][0] == seq[-1]:
            low = unity[i][1]
            high = unity[i][2]
    point = (low + high) / 2
    size_cod = math.ceil(math.log((1 / (high - low)), 2) + 1)
    bin_code = float_bin(point, size_cod)
    return [point, alphabet_size, alphabet, prob], bin_code


def decode_ac(encode_data_ac, sequence_length):
    point, alphabet_size, alphabet, prob = encode_data_ac
    unity = [[alphabet[i], sum(prob[:i]), sum(prob[:i + 1])] for i in range(alphabet_size)]
    decode_sequence = ""
    for i in range(sequence_length):
        for j in range(len(unity)):
            if unity[j][1] < point < unity[j][2]:
                symbol = unity[j][0]
                prob_low = unity[j][1]
                prob_high = unity[j][2]
                diff = prob_high - prob_low
                decode_sequence += symbol
                for k in range(len(unity)):
                    unity[k][1] = prob_low
                    unity[k][2] = prob[k] * diff + prob_low
                    prob_low = unity[k][2]
                break
    return decode_sequence


def encode_ch(uniq_chars, probabilitys, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[symbol] for symbol in alphabet]
    final = [[alphabet[i], probability[i]] for i in range(len(alphabet))]
    final.sort(key=lambda x: x[1])
    encode = ''
    symbol_code = []
    tree = []
    if 1 in probability and len(set(probability)) == 1:
        symbol_code = [[alphabet[i], "1" * i + "0"] for i in range(len(alphabet))]
        encode = "".join([symbol_code[alphabet.index(c)][1] for c in sequence])
    else:
        for i in range(len(final) - 1):
            i = 0
            left = final[i]
            final.pop(i)
            right = final[i]
            final.pop(i)
            tot = left[1] + right[1]
            tree.append([left[0], right[0]])
            final.append([left[0] + right[0], tot])
            final.sort(key=lambda x: x[1])
        symbol_code = []
        tree.reverse()
        alphabet.sort()
        for i in range(len(alphabet)):
            code = ""
            for j in range(len(tree)):
                if alphabet[i] in tree[j][0]:
                    code = code + '0'
                    if alphabet[i] == tree[j][0]:
                        break
                else:
                    code = code + '1'
                    if alphabet[i] == tree[j][1]:
                        break
            symbol_code.append([alphabet[i], code])
        encode = "".join([symbol_code[i][1] for i in range(len(alphabet)) if symbol_code[i][0] == c][0] for c in sequence)
    return [encode, symbol_code], encode


def decode_ch(encoded_sequence):
    sequence = ''
    encode = list(encoded_sequence[0])
    symbol_code = encoded_sequence[1]
    count = 0
    flag = 0

    for i in range(len(encode)):
        for j in range(len(symbol_code)):
            if encode[i] == symbol_code[j][1]:
                sequence += str(symbol_code[j][0])
                flag = 1

        if flag == 1:
            flag = 0
        else:
            count += 1

            if count == len(encode):
                break
            else:
                encode.insert(i + 1, str(encode[i] + encode[i + 1]))
                encode.pop(i + 2)

    return sequence

output = ''
results = []
with open('sequences.txt', 'r', encoding='utf-8') as f:
    for orig_sequence in f:
        sequence = orig_sequence[:10]
        sequence_len = len(sequence)
        unique_chars = set(sequence)
        sequence_alphabet_size = len(unique_chars)
        counts = collections.Counter(sequence)
        probability = {symbol: count / sequence_len for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())
        encoded_data_ac, encoded_sequence_ac = encode_ac(unique_chars, probability, sequence_alphabet_size, sequence)
        bps_ac = len(encoded_sequence_ac) / sequence_len
        decoded_sequence_ac = decode_ac(encoded_data_ac, sequence_len)
        encoded_data_ch, encoded_sequence_ch = encode_ch(unique_chars, probability, sequence)
        alph = ''
        for i in range(len(encoded_data_ch[1])):
            alph += f'{encoded_data_ch[1][i][0]}     {encoded_data_ch[1][i][1]}\n'
        bps_ch = len(encoded_sequence_ch) / sequence_len
        decoded_sequence_ch = decode_ch(encoded_data_ch)

        output += f'''Оригінальна послідовність: {sequence}
Ентропія: {entropy}

________________________________________Арифметичне кодування________________________________________
Дані закодованої АС послідовності: {encoded_data_ac} 
Закодована АС послідовність: {encoded_sequence_ac}
Значення bps при кодуванні АС: {bps_ac}
Декодована АС послідовність: {decoded_sequence_ac}

________________________________________Кодування Хаффмана________________________________________
Алфавіт Код символу
{alph}
Дані закодованої HС послідовності: {encoded_data_ch}
Закодована HС послідовність: {encoded_sequence_ch}
Значення bps при кодуванні HС: {bps_ch}
Декодована HС послідовність: {decoded_sequence_ch}

'''
        results.append([round(entropy, 3), bps_ac, bps_ch])

with open('results_AC_CH.txt', 'w', encoding='utf-8') as f:
    f.writelines(output)

fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
headers = ['Ентропія', 'bps AC', 'bps CH']
row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5','Послідовність 6', 'Послідовність 7', 'Послідовність 8']
ax.axis('off')
table = ax.table(cellText=results, colLabels=headers, rowLabels=row,loc='center', cellLoc='center')
table.auto_set_font_size(True)
table.set_fontsize(14)
table.scale(0.6, 2.2)
fig.savefig('Результати стиснення методами AC та CH' + '.jpg', dpi=300)
plt.close()