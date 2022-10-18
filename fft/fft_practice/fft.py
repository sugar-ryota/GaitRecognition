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

def fft_lowpassfillter(img,a=0.5):
    #フーリエ変換
    fimg = np.fft.fft2(img)
    fimg = np.fft.fftshift(fimg) #低周波成分を中央に寄せる
    #符号を保ったまま対数処理
    real = fimg.real
    # real[real>0] = np.log10(real[real>0])
    # real[real<0] = -np.log10(-real[real<0])
    imag = fimg.imag
    # imag[imag>0] = np.log10(imag[imag>0])
    # imag[imag<0] = -np.log10(-imag[imag<0])

    # 画像サイズ
    h, w = img.shape
   
    # 画像の中心座標
    cy, cx =  int(h/2), int(w/2)
    
    # フィルタのサイズ(矩形の高さと幅)
    rh, rw = int(a*cy), int(a*cx)

    # 入力画像と同じサイズで値0の配列を生成
    fdst = np.zeros(img.shape, dtype=complex)

    ffimg = np.zeros(real.shape, np.complex128)
    ffimg.real = real
    ffimg.imag = imag
    f = np.abs(ffimg)
    # 中心部分の値だけ代入（中心部分以外は0のまま）
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = f[cy-rh:cy+rh, cx-rw:cx+rw]
    
    # # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
    fdst =  np.fft.fftshift(fdst)

    f_real = fdst.real
    f_imag = fdst.imag

    # # 高速逆フーリエ変換 
    # dst = np.fft.ifft2(fdst)
   
    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    # return  np.uint8(dst.real)
    return f_real, f_imag

def fft_highpassfillter(src,a=0.5):
    # 高速フーリエ変換(2次元)
    src = np.fft.fft2(src)
    
    # 画像サイズ
    h, w = src.shape
   
    # 画像の中心座標
    cy, cx =  int(h/2), int(w/2)
    
    # フィルタのサイズ(矩形の高さと幅)
    rh, rw = int(a*cy), int(a*cx)

    # 第1象限と第3象限、第1象限と第4象限を入れ替え
    fsrc =  np.fft.fftshift(src)  

    # 入力画像と同じサイズで値0の配列を生成
    fdst = fsrc.copy()

    # 中心部分だけ0を代入（中心部分以外は元のまま）
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    
    # # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
    # fdst =  np.fft.fftshift(fdst)

    return fdst.real,fdst.imag


#単に高周波成分の重みを強くする関数
def highpass(img):
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
    return real,imag


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
img = cv2.imread("./fft/fft_practice/00000001.png",0)
show(img)

#フーリエ変換
# real, imag = FFT(img)
# real, imag = fft_lowpassfillter(img,0.3)
# real, imag = fft_highpassfillter(img,0.5)
real, imag = highpass(img)
showfft(real, imag)

#逆フーリエ変換
img = IFFT(real, imag)
show(img)