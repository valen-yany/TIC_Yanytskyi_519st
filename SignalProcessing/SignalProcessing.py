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
ax.set_xlabel('Частота (Гц) ', fontsize=14)
ax.set_ylabel('Амплітуда спектру', fontsize=14)

plt.title(f'Спектр сигналу з максимальною частотой F_max = {F_max} Гц', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'графік 2' + '.png', dpi=600)

discrete_signals = []
steps = (2, 4, 8, 16)
for Dt in steps:
    discrete_signal = numpy.zeros(n)
    for i in range(0, round(n/Dt)):
        discrete_signal[i*Dt] = filtered_signal[i*Dt]
    discrete_signals.append(list(discrete_signal))

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(timeline_ox, discrete_signals[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Час (секунди)', fontsize=14)
fig.supylabel('Амплітуда сигналу', fontsize=14)
fig.suptitle(f'Сигнал з кроком дискретизації Dt = {steps}', fontsize=14)
fig.savefig('./figures/' + 'графік 3' + '.png', dpi=600)

discrete_spectrums = []
for Ds in discrete_signals:
    spectrum = fft.fft(Ds)
    spectrum = numpy.abs(fft.fftshift(spectrum))
    discrete_spectrums.append(list(spectrum))

freq_countdown = fft.fftfreq(n, 1/n)
freq_countdown = fft.fftshift(freq_countdown)
fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(freq_countdown, discrete_spectrums[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Частота (Гц) ', fontsize=14)
fig.supylabel('Амплітуда спектру', fontsize=14)
fig.suptitle(f'Спектри сигналів з кроком дискретизації Dt = {steps}', fontsize=14)
fig.savefig('./figures/' + 'графік 4' + '.png', dpi=600)

F_filter = 42
w = F_filter/(Fs/2)
parameters_filter = signal.butter(3, w, 'low', output='sos')
filtered_discretes_signal = []
for discrete_signal in discrete_signals:
    discrete_signal = signal.sosfiltfilt(parameters_filter, discrete_signal)
    filtered_discretes_signal.append(list(discrete_signal))

fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))
s = 0
for i in range(0, 2):
    for j in range(0, 2):
        ax[i][j].plot(timeline_ox, filtered_discretes_signal[s], linewidth=1)
        ax[i][j].grid()
        s += 1
fig.supxlabel('Час (секунди)', fontsize=14)
fig.supylabel('Амплітуда сигналу', fontsize=14)
fig.suptitle(f'Відновлені аналогові сигнали з кроком дискретизації Dt = {steps}', fontsize=14)
fig.savefig('./figures/' + 'графік 5' + '.png', dpi=600)

dispersions = []
signal_noise = []
for i in range(len(steps)):
    E1 = filtered_discretes_signal[i] - filtered_signal
    dispersion = numpy.var(E1)
    dispersions.append(dispersion)
    signal_noise.append(numpy.var(filtered_signal)/dispersion)

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(steps, dispersions, linewidth=1)
ax.set_xlabel('Крок дискретизації', fontsize=14)
ax.set_ylabel('Дисперсія', fontsize=14)
plt.title(f'Залежність дисперсії від кроку дискретизації', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'графік 6' + '.png', dpi=600)

fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
ax.plot(steps, signal_noise, linewidth=1)
ax.set_xlabel('Крок дискретизації', fontsize=14)
ax.set_ylabel('ССШ', fontsize=14)
plt.title(f'Залежність співвідношення сигнал-шум від кроку дискретизації', fontsize=14)
ax.grid()
fig.savefig('./figures/' + 'графік 7' + '.png', dpi=600)