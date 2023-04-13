import numpy
from scipy import signal, fft
import matplotlib.pyplot as plt

n = 500
Fs = 1000
F_max = 35

random = numpy.random.normal(0, 10, n)

timeline_ox = numpy.arange(n)/Fs

w = F_max/(Fs/2)
parameters_filter = signal.butter(3, w, 'low', output='sos')
filtered_signal = signal.sosfiltfilt(parameters_filter, random)

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(timeline_ox, filtered_signal, linewidth=1)
ax.set_xlabel('Час(секунди)', fontsize=14)
ax.set_ylabel('Амплітуда сигналу', fontsize=14)
plt.title(f'Сигнал з максимальною частотою F_max = {F_max} Гц', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'графік 1' + '.png', dpi=600)

spectrum = fft.fft(filtered_signal)
spectrum = numpy.abs(fft.fftshift(spectrum))
length_signal = n
freq_countdown = fft.fftfreq(length_signal, 1/length_signal)
freq_countdown = fft.fftshift(freq_countdown)

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(freq_countdown, spectrum, linewidth=1)
ax.set_xlabel('Час (секунди) ', fontsize=14)
ax.set_ylabel('Амплітуда спектру', fontsize=14)

plt.title(f'Спектр сигналу з максимальною частотой F_max = {F_max} Гц', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'графік 2' + '.png', dpi=600)
