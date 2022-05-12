# %%
import numpy as np
import matplotlib.pyplot as plt

# データの取得時、デジタルデータは一定の間隔で取得される。
# 元の信号のサンプリング周波数は1/dt(つまり1000Hz)
N = 1024            # サンプル数 (サンプリングする数)
dt = 0.001          # サンプリング周期 [s] 0.001秒間隔でデータをサンプリング (何秒に1回データを取得しているか、その間隔)
f1, f2 = 50, 120    # 周波数 [Hz] (1秒間に繰り返す波の数)

# N*dt =　サンプル数だけデータを取得するのにかかる時間
t = np.arange(0, N*dt, dt)  # 時間 [s]
x = 1.5*np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 3  # 信号

# 横軸は時間を表し、間隔は0.001 (つまり、1024分割されてる)
# 縦軸は信号を表している
fig, ax = plt.subplots()
ax.plot(t, x)
# ax.set_xlim(0, 0.1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
ax.grid()
plt.show()

F = np.fft.fft(x, n=N)  # フーリエ変換結果(戻り値は長さNの複素数配列(2次元))
freq = np.fft.fftfreq(N, d=dt)  # 周波数を取得,dはサンプリング周期
freq_array = np.array(freq)

fig, ax = plt.subplots()
ax.plot(F.real, label='real_part')
ax.legend()
# ax.set_xlim(0, 0.1)
ax.set_xlabel("Number of data")
ax.set_ylabel("real_part_power")
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.plot(freq, label='frequency')  # frequencyは周波数
# ax.set_xlim(0, 0.1)
ax.legend()
ax.set_xlabel("Number of data")
ax.set_ylabel("frequency")
ax.grid()
plt.show()

# fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 6))
# ax[0].plot(F.real, label="Real part") #横軸は時間、縦軸はスペクトルの実部
# ax[0].legend()
# ax[1].plot(F.imag, label="Imaginary part")
# ax[1].legend()
# ax[2].plot(freq, label="Frequency")
# ax[2].legend()
# ax[2].set_xlabel("Number of data")
# plt.show()

Amp = np.abs(F/(N/2))  # 振幅

fig, ax = plt.subplots()
ax.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
ax.set_xlabel("Freqency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()

# %%
