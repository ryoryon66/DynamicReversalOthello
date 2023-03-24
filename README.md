
# DynamicReversalOthello
DynamicReversalOthelloは、Pythonを使用した古典的なボードゲーム「オセロ」（別名：リバーシ）の変則バージョンの実装です。tkinterライブラリを使用しています。

基本的には通常のオセロと同じですが得点の計算方法が違います。
設定ファイル(othelloRL/settings.py)で指定されたスコア計算方法がpow1の時は、
毎ターンひっくり返した駒の数の分だけ点数が入ります。pow2が指定されている場合は、ひっくり返した駒の数の二乗分だけ点数が入ります。
両者ともに駒を置くことができなくなったときに、スコアの合計が大きかった方の勝ちです。

人間側には一手ごとに時間制限があります。


# 使い方

```
git clone git@github.com:ryoryon66/DynamicReversalOthello.git
```



以下の手順でプログラムを実行します。

必要なライブラリをインストールします。
```
pip install numpy
pip install torch
```


このリポジトリのソースコードをダウンロードし、vs_cpu_gui.pyを実行します。

ゲームウィンドウが表示されたら、人間プレイヤーが白、CPUプレイヤーが黒です。各プレイヤーは交互に石を置いていきます。石を置くことが可能な場所がハイライトされるので、その場所をクリックしてください。
先手はCPUに固定されています。

時間制限内に手を打たないと、人間は敗北します。制限時間は設定ファイル(othelloRL/settings.py)で変更できます。

ゲームが終了したら、勝者が表示されます。すべての石が盤上に置かれるか、どちらのプレイヤーも合法手がなくなるとゲームが終了します。

# 注意事項
このプログラムは、オセロの基本ルールに従っています。合法手がないプレイヤーはパスする必要があります。

CPUプレイヤーは、DDQN（Deep Double Q-Network）エージェントを使用しています。エージェントは、学習済みの重みファイルを使用しています.
使用する重みファイルも設定ファイル(othelloRL/settings.py)から設定できます。

experiment2/ddqn_best_weights.pthのランダムエージェ
ントに対する勝率はおよそ85%
pow2_experiment2/ddqn_best_weights.pthのランダムエージェントに対する勝率はおよそ91.5%

