import os
from huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image


def dct_2d(image):
	return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def quantize(block, component, table_num):
	q = load_quantization_table(component, table_num)
	return (block / q).round().astype(np.int32)


def load_quantization_table(component, table_num):
	q = 0
	if component == 'lum':
		if table_num == 1:
			q = np.array([
				[2, 2, 2, 2, 3, 4, 5, 6],
				[2, 2, 2, 2, 3, 4, 5, 6],
				[2, 2, 2, 2, 4, 5, 7, 9],
				[2, 2, 2, 4, 5, 7, 9, 12],
				[3, 3, 4, 5, 8, 10, 12, 12],
				[4, 4, 5, 7, 10, 12, 12, 12],
				[5, 5, 7, 9, 12, 12, 12, 12],
				[6, 6, 9, 12, 12, 12, 12, 12]])
		elif table_num == 2:
			q = np.array([
				[16, 11, 10, 16, 24, 40, 51, 61],
				[12, 12, 14, 19, 26, 48, 60, 55],
				[14, 13, 16, 24, 40, 57, 69, 56],
				[14, 17, 22, 29, 51, 87, 80, 62],
				[18, 22, 37, 56, 68, 109, 103, 77],
				[24, 35, 55, 64, 81, 104, 113, 92],
				[49, 64, 78, 87, 103, 121, 120, 101],
				[72, 92, 95, 98, 112, 100, 103, 99]])
	elif component == 'chrom':
		if table_num == 1:
			q = np.array([
				[3, 3, 5, 9, 13, 15, 15, 15],
				[3, 4, 6, 11, 14, 12, 12, 12],
				[5, 6, 9, 14, 12, 12, 12, 12],
				[9, 11, 14, 12, 12, 12, 12, 12],
				[13, 14, 12, 12, 12, 12, 12, 12],
				[15, 12, 12, 12, 12, 12, 12, 12],
				[15, 12, 12, 12, 12, 12, 12, 12],
				[15, 12, 12, 12, 12, 12, 12, 12]])
		elif table_num == 2:
			q = np.array([
				[17, 18, 24, 47, 99, 99, 99, 99],
				[18, 21, 26, 66, 99, 99, 99, 99],
				[24, 26, 56, 99, 99, 99, 99, 99],
				[47, 66, 99, 99, 99, 99, 99, 99],
				[99, 99, 99, 99, 99, 99, 99, 99],
				[99, 99, 99, 99, 99, 99, 99, 99],
				[99, 99, 99, 99, 99, 99, 99, 99],
				[99, 99, 99, 99, 99, 99, 99, 99]])

	else:
		raise ValueError(f"component must be \"lum\" or \"chrom\", but {component} exist")
	return q


def zigzag_points(rows, cols):
	UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

	def move(direction, point):
		return {
			UP: lambda point: (point[0] - 1, point[1]),
			DOWN: lambda point: (point[0] + 1, point[1]),
			LEFT: lambda point: (point[0], point[1] - 1),
			RIGHT: lambda point: (point[0], point[1] + 1),
			UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
			DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
		}[direction](point)

	def inbounds(point):
		return 0 <= point[0] < rows and 0 <= point[1] < cols

	point = (0, 0)

	move_up = True
	for i in range(rows * cols):
		yield point
		if move_up:
			if inbounds(move(UP_RIGHT, point)):
				point = move(UP_RIGHT, point)
			else:
				move_up = False
				if inbounds(move(RIGHT, point)):
					point = move(RIGHT, point)
				else:
					point = move(DOWN, point)
		else:
			if inbounds(move(DOWN_LEFT, point)):
				point = move(DOWN_LEFT, point)
			else:
				move_up = True
				if inbounds(move(DOWN, point)):
					point = move(DOWN, point)
				else:
					point = move(RIGHT, point)


def block_to_zigzag(block):
	return np.array([block[point] for point in zigzag_points(*block.shape)])


def bits_required(n):
	n, result = abs(n), 0
	while n > 0:
		n >>= 1
		result += 1
	return result


def flatten(lst):
	return [itm for sublst in lst for itm in sublst]


def run_length_encode(arr):
	last_nonzero = -1
	run_length = 0

	for i, elem in enumerate(arr):
		if elem != 0:
			last_nonzero = i

	symbols = []
	values = []

	for i, elem in enumerate(arr):
		if i > last_nonzero:
			symbols.append((0, 0))
			values.append(int_to_binstr(0))
			break
		elif elem == 0 and run_length < 15:
			run_length += 1
		else:
			size = bits_required(elem)
			symbols.append((run_length, size))
			values.append(int_to_binstr(elem))
			run_length = 0

	return symbols, values


def bin_string_flip(bin_str):
	if not set(bin_str).issubset('01'):
		raise ValueError("bin_string must contain only '0' or '1'")
	return ''.join(map(lambda c: '0' if c == '1' else '1', bin_str))


def uint_to_binstr(number, size):
	return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(number):
	if number == 0:
		return ''
	binary_string = bin(abs(number))[2:]
	return binary_string if number > 0 else bin_string_flip(binary_string)


def write_to_file(filepath, dc, ac, blocks_count, tables):
	file = open(filepath, 'w')
	for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
		file.write(uint_to_binstr(len(tables[table_name]), 16))
		for key, value in tables[table_name].items():
			if table_name in {'dc_y', 'dc_c'}:
				file.write(uint_to_binstr(key, 4))
				file.write(uint_to_binstr(len(value), 4))
				file.write(value)
			else:
				file.write(uint_to_binstr(key[0], 4))
				file.write(uint_to_binstr(key[1], 4))
				file.write(uint_to_binstr(len(value), 8))
				file.write(value)

	file.write(uint_to_binstr(blocks_count, 32))
	for b in range(blocks_count):
		for c in range(3):
			cat = bits_required(dc[b, c])
			sym, val = run_length_encode(ac[b, :, c])
			dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
			ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
			file.write(dc_table[cat])
			file.write(int_to_binstr(dc[b, c]))
			for i in range(len(sym)):
				file.write(ac_table[tuple(sym[i])])
				file.write(val[i])
	file.close()


def encode_image(output_file_name: str, input_file_name: str, table_number: int):
	input_file_path = f"{input_file_name}.bmp"
	output_file_path = f"{output_file_name}.asf"
	block_index = 0
	image = Image.open(input_file_path)
	ycbcr_image = image.convert('YCbCr')
	np_image = np.array(ycbcr_image, dtype=np.uint8)

	rows, cols = np_image.shape[0], np_image.shape[1]

	if rows % 8 == cols % 8 == 0:
		num_blocks = rows // 8 * cols // 8
	else:
		raise ValueError("Image size must be divisible by 8")

	dc = np.empty((num_blocks, 3), dtype=np.int32)
	ac = np.empty((num_blocks, 63, 3), dtype=np.int32)

	for i in range(0, rows, 8):
		for j in range(0, cols, 8):
			try:
				block_index += 1
			except NameError:
				block_index = 0

			for k in range(3):
				block = np_image[i:i + 8, j:j + 8, k] - 128
				dct_matrix = dct_2d(block)
				quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom', table_number)
				zigzag = block_to_zigzag(quant_matrix)
				dc[block_index, k] = zigzag[0]
				ac[block_index, :, k] = zigzag[1:]

	huffman_dc_y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
	huffman_dc_c = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
	huffman_ac_y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0] for i in range(num_blocks)))
	huffman_ac_c = HuffmanTree(flatten(run_length_encode(ac[i, :, j])[0] for i in range(num_blocks) for j in [1, 2]))

	tables = {
		'dc_y': huffman_dc_y.value_to_bitstring_table(),
		'ac_y': huffman_ac_y.value_to_bitstring_table(),
		'dc_c': huffman_dc_c.value_to_bitstring_table(),
		'ac_c': huffman_ac_c.value_to_bitstring_table()
	}

	input_file_size = os.path.getsize(input_file_path)

	with open("results_jpeg.txt", "a", encoding="utf8") as file:
		print(f'Quantization table - {table_number}, Image - {input_file_name}', file=file)
		print(f'Original file size: {input_file_size} bytes', file=file)

	write_to_file(f"results/{output_file_path}", dc, ac, num_blocks, tables)

	return input_file_size


class JPEGFileReader:

	def __init__(self, filepath):
		self.file = open(filepath, 'r')

	def read_int(self, size):
		if size == 0:
			return 0
		bin_num = self.read_str(size)
		if bin_num[0] == '1':
			return self.int2(bin_num)
		else:
			return self.int2(self.binstr_flip(bin_num)) * -1

	def read_dc_table(self):
		table = dict()
		table_size = self.read_uint(16)
		for _ in range(table_size):
			category = self.read_uint(4)
			code_length = self.read_uint(4)
			code = self.read_str(code_length)
			table[code] = category
		return table

	def read_ac_table(self):
		table = dict()
		table_size = self.read_uint(16)
		for _ in range(table_size):
			run_length = self.read_uint(4)
			size = self.read_uint(4)
			code_length = self.read_uint(8)
			code = self.read_str(code_length)
			table[code] = (run_length, size)
		return table

	def read_blocks_count(self):
		return self.read_uint(32)

	def read_huffman_code(self, table):
		prefix = ''
		while prefix not in table:
			char = self.read_char()
			if char:
				prefix += char
			else:
				break
		return table[prefix]

	def read_uint(self, size):
		if size <= 0:
			raise ValueError("розмір повинен бути більшим за 0")
		return self.int2(self.read_str(size))

	def read_str(self, length):
		return self.file.read(length)

	def read_char(self):
		return self.read_str(1)

	def int2(self, bin_num):
		return int(bin_num, 2)

	def binstr_flip(self, bin_str):
		return bin_str[::-1]


def decoder(output_f: str, input_f: str, table_num: int, size_f: int):
	input_file = f"results/{input_f}.asf"
	output_file = f"results/{output_f}.jpeg"
	dc, ac, tables, blocks_count = read_image_file(input_file)
	block_side = 8
	image_side = int(math.sqrt(blocks_count)) * block_side
	blocks_per_line = image_side // block_side
	npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)
	for block_index in range(blocks_count):
		i = block_index // blocks_per_line * block_side
		j = block_index % blocks_per_line * block_side
		for c in range(3):
			zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
			quant_matrix = zigzag_to_block(zigzag)
			dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom', table_num)
			block = idct_2d(dct_matrix)
			npmat[i:i + 8, j:j + 8, c] = block + 128
	image = Image.fromarray(npmat, 'YCbCr')
	image = image.convert('RGB')
	image.save(output_file)
	size_jpeg = os.path.getsize(output_file)
	width, height = image.size
	ratio = size_f / size_jpeg
	with open("results_jpeg.txt", "a", encoding="utf8") as file:
		print(f'Розмір файла JPEG: {size_jpeg} байт', file=file)
		print(f'Розмір зображення JPEG: {width}x{height}', file=file)
		print(f'Коефіцієнт стиснення = {ratio}\n', file=file)


def read_image_file(filepath):
	reader = JPEGFileReader(filepath)
	tables = dict()
	for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
		tables[table_name] = reader.read_dc_table() if 'dc' in table_name else reader.read_ac_table()
	blocks_count = reader.read_blocks_count()
	dc = np.empty((blocks_count, 3), dtype=np.int32)
	ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
	for block_index in range(blocks_count):
		for component in range(3):
			dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
			ac_table = tables['ac_y'] if component == 0 else tables['ac_c']
			category = reader.read_huffman_code(dc_table)
			dc[block_index, component] = reader.read_int(category)
			cells_count = 0
			while cells_count < 63:
				run_length, size = reader.read_huffman_code(ac_table)
				if (run_length, size) == (0, 0):
					while cells_count < 63:
						ac[block_index, cells_count, component] = 0
						cells_count += 1
				else:
					for _ in range(run_length):
						ac[block_index, cells_count, component] = 0
						cells_count += 1
					if size == 0:
						ac[block_index, cells_count, component] = 0
					else:
						value = reader.read_int(size)
						ac[block_index, cells_count, component] = value
					cells_count += 1
	return dc, ac, tables, blocks_count


def zigzag_to_block(zigzag):
	rows = cols = int(math.sqrt(len(zigzag)))
	if rows * cols != len(zigzag):
		raise ValueError("Length must be ideal square")
	block = np.empty((rows, cols), np.int32)
	for i, point in enumerate(zigzag_points(rows, cols)):
		block[point] = zigzag[i]
	return block


def dequantize(block, component, table_num):
	q = load_quantization_table(component, table_num)
	return block * q


def idct_2d(image):
	return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


files = ["17_1", "17_2", "17_3"]
for i in range(1, 3):
	for j in files:
		size = encode_image(f"{j}_{i}", j, i)
		decoder(f"{j}_{i}", f"{j}_{i}", i, size)