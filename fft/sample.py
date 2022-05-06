import numpy as np
import matplotlib.pyplot as plt
import cv2

#フーリエ変換
def FFT(img):
    #フーリエ変換
    fimg = np.fft.fft2(img)
    fimg = np.fft.fftshift(fimg) #低周波成分を中央に寄せる
    #符号を保ったまま対数処理
    real = fimg.real
    real[real>0] = np.log10(real[real>0])
    real[real<0] = -np.log10(-real[real<0])
    imag = fimg.imag
    imag[imag>0] = np.log10(imag[imag>0])
    imag[imag<0] = -np.log10(-imag[imag<0])
    return real, imag

#逆フーリエ変換
def IFFT(real, imag):
    #符号を保ったまま指数処理
    real[real>0] = 10**real[real>0]
    real[real<0] = -10**(-real[real<0])
    imag[imag>0] = 10**imag[imag>0]
    imag[imag<0] = -10**(-imag[imag<0])
    #複素数行列
    fimg = np.zeros(real.shape, np.complex128)
    fimg.real = real
    fimg.imag = imag
    #逆フーリエ変換
    fimg = np.fft.ifftshift(fimg)
    img = np.fft.ifft2(fimg)
    img = img.real
    img = img.clip(0, 255).astype(np.uint8)
    return img


#画像の表示関数
def show(img):
    plt.figure()
    plt.imshow(img)
    plt.gray()
    plt.show()

#FFT画像の表示関数
def showfft(real, imag):
    fimg = np.zeros(real.shape, np.complex128)
    fimg.real = real
    fimg.imag = imag
    img = np.abs(fimg)
    plt.figure()
    plt.imshow(img)
    plt.gray()
    plt.show()


#グレースケール読み込み
img = cv2.imread('data/TreadmillDatasetA/00001/gallery_2km/00000001.png',0) #0はグレースケール
show(img)

#フーリエ変換
real, imag = FFT(img)
showfft(real, imag)
# print(real)

#逆フーリエ変換
img = IFFT(real, imag)
show(img)