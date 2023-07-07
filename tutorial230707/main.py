'''
    Python の基本機能
'''
a = 1       # aに1を代入
print(a)    # aを表示
print(type(a))

b = 1.0
print(b)
print(type(b))

c = 1.
print(c,type(c)) # 変数と型の表示

# 関数定義（掛け算）
def product( left, right):
    answer = left * right
    return answer

d = product( a, 3.) # 関数を使うとき
print(d,type(d))

for i in range(1,5):    # iが1～4までループ
    print(i,type(i))

if b%2 == 0:
    print(b,"偶数です")
else:
    print(b,"奇数です")

for i in range(1,5):    # iが1～4までループ
    b = b+i # bにiを加算
    if b%2==0 :
        print(b,"偶数です")
    else:
        print(b,"奇数です")


'''
    NumPy （数値計算ライブラリ）の機能
'''
import numpy as np  # numpyをnpとしてインポート
print(np.pi) # 円周率

li = list(range(1,5))   # list型
print(li,type(li))

vr = np.array([1., 2., 3.]) # 行ベクトルの宣言
print(vr,type(vr))

vc = np.array([[1.],[2.],[3.]]) # 列ベクトルの宣言
print(vc,type(vc))

# 行列の宣言
A = np.array([[ 1., 2., 3. ],
              [ 0., 4., 5. ],
              [ 0., 0., 6. ]])
print(A,type(A))

# 行列とベクトルの積
bc = A @ vc
print(bc,type(bc))

# ar = 1:0.1:4.9
ar = np.arange(1,5,0.1) # 1から5未満までの0.1間隔の配列
print(ar,type(ar))

ar1 = ar[0] # arの最初の要素をar1に代入
print(ar1,type(ar1))

ar_end = ar[-1] # arの最後の要素にアクセス
print(ar_end,type(ar_end))

ar_end_1 = ar[-2] # arの最後から1つ前の要素にアクセス
print(ar_end_1,type(ar_end_1))

ar_part = ar[0:5]   # 配列の一部取り出し（スライス）
print(ar_part,type(ar_part))

ar_part_last = ar[-5:len(ar)]   # 配列の最後の一部のスライス
print(ar_part_last,type(ar_part_last))

A_inv = np.linalg.inv(A) # 逆行列の計算
print(A_inv,type(A_inv))


'''
    PyTorch（機械学習ライブラリの1つ） の使用
'''
import torch    # PyTorchのインポート

a = torch.tensor(2.)    # 2.をTensor型で宣言
print(a,type(a))

# Tensor型で宣言（requires_gradをオン，勾配情報を有効化）
x = torch.tensor(5., requires_grad=True)
print(x,type(x))

# Xについての非線形関数を定義
#  Y = A * X^2
def nonlinear_model( X ):
    A = 2.
    Y = A * X ** 2
    return Y

y = nonlinear_model( x )    # 非線形関数にxを代入（順方向計算）
print(y,type(y))

y.backward()    # 逆伝搬を実行
print(x.grad,type(x.grad))  # 勾配を表示
