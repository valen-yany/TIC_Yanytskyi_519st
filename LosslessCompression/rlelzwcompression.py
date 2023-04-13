import collections
import math
from matplotlib import pyplot as plt

def rle_encode(sequence):
    sym = sequence[0]
    count = 1
    result = []
    encoded = ''
    for i, letter in enumerate(sequence[1:]):
        if letter != sym:
            result.append((count, sym))
            encoded += str(count) + sym
            count = 1
            sym = letter
            continue
        count += 1
    result.append((count, sym))
    encoded += str(count) + sym
    return encoded, result


def rle_decode(rle_encode):
    result = ''
    for count, letter in rle_encode:
        result += letter * count
    return result.strip('\n')


def lzw_encode(sequence):
    dictionary = {}
    for i in range(65536):
        dictionary[chr(i)] = i
    output = ''
    current = ""
    r = []
    s = 0
    for char in sequence:
        n = current + char
        if n in dictionary:
            current = n
        else:
            r.append(dictionary[current])
            dictionary[n] = len(dictionary)
            b = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
            current = char
            output += f"Code: {dictionary[current]}, Element: {current}, Bits: {b}\n"
            s = s + b
    l = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
    s = s + l
    output += f"Code: {dictionary[current]}, Element: {current}, Bits: {l}\n"
    r.append(dictionary[current])
    return r, s, output


def lzw_decode(sequence):
    d = {}
    for i in range(65536):
        d[i] = chr(i)
    r = ""
    p = None
    c = ""
    for code in sequence:
        if code in d:
            c = d[code]
            r += c
            if p is not None:
                d[len(d)] = p + c[0]
            p = c
        else:
            c = p + p[0]
            r += c
            d[len(d)] = c
            p = c
    return r


def main():
    N_sequence = 100
    original_sequence_size = N_sequence * 16
    original_sequences = []
    outputs = []
    results = []
    with open('sequences.txt', 'r') as f:
        for line in f:
            original_sequences.append(line.strip('\n'))
    for sequence in original_sequences:
        output = f''''''
        counts = collections.Counter(sequence)
        probability = {symbol: count / N_sequence for symbol, count in counts.items()}
        entropy = round(-sum(p * math.log2(p) for p in probability.values()), 2)
        output += f'''///////////////////////////////
Оригінальна послідовність: {sequence}
Розмір оригінальної послідовності: {original_sequence_size} bits
Ентропія: {entropy}

'''
        rle_sequence, rle_encoded = rle_encode(sequence)
        rle_sequence_size = len(rle_sequence) * 16
        cr_rle = round(original_sequence_size / rle_sequence_size, 3)
        if cr_rle < 1:
            cr_rle = '-'
        decoded_rle = rle_decode(rle_encoded)
        output += f'''_____Кодування_RLE_____
Закодована RLE послідовність : {rle_sequence} 
Розмір закодованої RLE послідовності: {rle_sequence_size} bits
Коефіцієнт стиснення RLE: {cr_rle}
Декодована RLE послідовність: {decoded_rle}
Розмір декодованої RLE послідовності: {len(decoded_rle) * 16} bits

'''
        encoded_lzw, size, lzw_output = lzw_encode(sequence)
        cr_lzw = round((original_sequence_size / size), 2)
        if cr_lzw < 1:
            cr_lzw = '-'
        decoded_result_lzw = lzw_decode(encoded_lzw)
        output += f'''_____Кодування_LZW_____
_____Поетапне_кодування_____
{lzw_output}
____________________________
Закодована LZW послідовність:{''.join(map(str, encoded_lzw))}
Розмір закодованої LZW послідовності: {size} bits
Декодована LZW послідовність:{decoded_result_lzw}
Розмір декодованої LZW послідовності: {len(decoded_result_lzw) * 16} bits

'''
        results.append([entropy, cr_rle, cr_lzw])
        outputs.append(output)
    N = len(original_sequences)
    fig, ax = plt.subplots(figsize=(14 / 1.54, N / 1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    row = [f'Послідовність {i}' for i in range(1, N + 1)]
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    plt.savefig('Результати стиснення методами RLE та LZW' + '.png', dpi=600)
    plt.show()
    with open('results_rle_lzw.txt', 'w') as f:
        for i in outputs:
            f.writelines(i)


if __name__ == '__main__':
    main()
