'''
    1. パッケージのインポート
'''
import numpy as np  # NumPy（数値計算パッケージ）のインポート
import matplotlib.pyplot as plt # 描画パッケージのインポート
'''
    2. 学習用データ点の生成（or 読み込み）
'''
x_data = np.arange(0, 5.1, 0.1) # 0～5までの0.1間隔の配列
print(x_data,type(x_data))  # データの表示
# データ生成関数
#  y = sin(2(x+3.5)) / (x+3.5) + 0.1x + 0.2*[一様乱数（0~1）]
y_data =np.sin(2*(x_data+3.5))/(x_data+3.5) + 0.1 * x_data \
        + 0.2 * np.random.rand(len(x_data))
print(y_data,type(y_data))

'''
    3. 学習モデルの生成
'''
# 基本形の関数（カーネル関数）を定義
def kernel_func(x,c):   # x:入力，c:中心
    h = 0.3 # バンド幅
    y = np.exp(-0.5*(x-c)**2/h**2)  # ガウシアンカーネル
    return y
# 学習モデルを生成する高階関数
def genLearningModel(x_tick): # 学習用データのx座標の配列を与える
    # 学習モデル関数の定義
    def learning_model(x,w):
        y = 0   # 出力の初期化
        for i in range(len(w)): # 重み配列の長さ分の繰り返し文
            y = y +w[i] * kernel_func(x,x_tick[i])
        return y
    return learning_model

# 学習モデルの生成
learning_model1 = genLearningModel(x_data.copy())    # x_dataを値渡し
# learning_model2 = genLearningModel(x_data.copy())    # x_dataを値渡し
# 作成した学習モデルの試用
y_hat1 = learning_model1(x_data, np.random.rand(len(x_data)))
# y_hat2 = learning_model2(x_data, np.random.rand(len(x_data)))

'''
    4. 学習
'''
'''
    5. 図の出力
'''
plt.figure("回帰問題の例題") # 図の宣言

plt.subplot(211)    # サブプロットの作成（2行1列のうち1番目）
# 学習用データの表示
plt.plot( x_data, y_data, 'o', label="Original data")
# 近似データの表示
plt.plot( x_data, y_hat1, ':', label="Regression model1")
# plt.plot( x_data, y_hat2, ':', label="Regression model2")
# 表示オプションの設定
plt.xlim( x_data[0], x_data[-1])    # x軸の範囲設定
plt.grid()      # グリッドを表示
plt.legend()    # 凡例を表示
plt.xlabel("Time [step]")  # x軸のラベルを設定
plt.ylabel("Output [-]")   # y軸のラベルを設定

# 図の表示
plt.show()
