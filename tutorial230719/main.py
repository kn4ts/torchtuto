'''
    1. パッケージのインポート
'''
import numpy as np  # NumPy（数値計算パッケージ）のインポート
import matplotlib.pyplot as plt # 描画パッケージのインポート
import torch

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
y_hat1 = learning_model1(x_data, np.linspace(0,1,len(x_data)))
#y_hat1 = learning_model1(x_data, np.random.rand(len(x_data)))
# y_hat2 = learning_model2(x_data, np.random.rand(len(x_data)))

'''
    4. 学習
'''
# 損失関数の定義（学習データとモデル出力の誤差を評価する式）
def loss_func( y, y_hat ):
    J = (y-y_hat)**2    # 誤差の2乗
    return J
# 確率的勾配降下（SGD）法の繰り返し学習の関数
def training_loop_func(n_epochs, eta, w, x_data, y_data):
    J_h = np.zeros(n_epochs)    # 損失関数値の保存用配列
    for epoch in range(n_epochs):
        # 重みのgrad属性が0でないときにgradを0にリセット
        if w.grad is not None:
            w.grad.zero_()
        # 学習データから1つをランダムに選択
        j = np.random.randint(0,len(x_data))
        # 学習モデル出力の計算
        y_hat = learning_model1(x_data[j],w)
        # 損失関数値の計算
        J = loss_func(y_data[j],y_hat)
        # 逆伝搬の実行
        J.backward()
        # 勾配法によるパラメータ更新
        with torch.no_grad():   # with文を使ってno_gradコンテキストで更新部分をカプセル化
            w -= eta * w.grad # autogradで順伝搬のグラフにエッジが追加されないようにしている

        # 途中経過の表示
        if epoch % 100 == 0: # epoch100回ごとに表示
            print('Epoch %d, J= %f' % (epoch, float(J)))
        # 損失関数値の系列を保存
        J_h[epoch] = J

    return w, J_h

# 重みの初期値の設定
w0 = np.array(np.linspace(0,1,len(x_data))) # 0~1の等差数列
w = torch.tensor(w0, requires_grad=True) # np.arrayをtensor化

eta = 0.01 # 学習率の定義

# SGDによる繰り返し学習
w, J_h = training_loop_func(1000, eta, w, x_data, y_data)

# 初期値と1000回更新後の回帰モデルの比較
y_hat0 = learning_model1(x_data,w0)  # 初期重みによるモデル出力
y_hat1 = learning_model1(x_data,w.detach().numpy().copy()) # 1000回更新後

## 学習データから1つ選ぶ
#j = 30 # jは学習データのインデックス
## 選択したデータのx(と重みw)からモデル出力を計算
#y_hat = learning_model1( x_data[j], w)
#print(y_hat,type(y_hat))
## 損失関数値を計算
#J = loss_func(y_data[j], y_hat)
#print(J,type(J))
## 勾配を計算
#J.backward() # 逆伝搬の実行
#print("自動微分で計算した勾配",w.grad, type(w.grad)) # 勾配の表示
#
## 勾配を使った重みの更新
#eta = 0.1 # 学習率
#w = w - eta * w.grad    # 重みの更新

#''' 4.5 勾配の検算 '''
#def dJdw(x_data,y_data,w,j): # 勾配を手計算で求める関数
#                             # 戻り値：勾配ベクトル
#    v = np.zeros(len(w)) # 重みと同じ長さの空のベクトル
#    for i in range(len(w)):
#        v[i] = -2. * kernel_func(x_data[j],x_data[i]) \
#                * ( y_data[j] -learning_model1(x_data[j],w))
#    return v
#jw = dJdw(x_data,y_data,w,j)
#print("手計算で計算した勾配",jw,type(jw))

'''
    5. 図の出力
'''
plt.figure("回帰問題の例題") # 図の宣言

plt.subplot(211)    # サブプロットの作成（2行1列のうち1番目）
# 学習用データの表示
plt.plot( x_data, y_data, 'o', label="Original data")
# 近似データの表示
plt.plot( x_data, y_hat0, ':', label="Initial regression model")
plt.plot( x_data, y_hat1, ':', label="Trained regression model")
# 表示オプションの設定
plt.xlim( x_data[0], x_data[-1])    # x軸の範囲設定
plt.ylim( 0., 1.)    # y軸の範囲設定
plt.grid()      # グリッドを表示
plt.legend()    # 凡例を表示
plt.xlabel("Time [step]")  # x軸のラベルを設定
plt.ylabel("Output [-]")   # y軸のラベルを設定

plt.subplot(212)
# エポックごとの損失関数値の表示
plt.plot( range(len(J_h)), J_h, '-', \
         label="Loss function value" )
plt.yscale('log')   # y軸を対数スケールに
plt.xlim( 0, len(J_h))    # x軸の範囲設定
plt.grid()      # グリッドを表示
plt.legend()    # 凡例を表示
plt.xlabel("Epoch number [-]")  # x軸のラベルを設定
plt.ylabel("Loss function value [-]")   # y軸のラベルを設定

# 図の表示
plt.show()
