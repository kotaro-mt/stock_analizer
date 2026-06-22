# stock_future — 自己改善ループ指示書 (Claude 向け)

## ミッション (Phase A 以降)
約 100 銘柄の東証プライム株について、過去 60 営業日の特徴量から
「次の 20 営業日の **二重中心化残差** 日次 log リターン」
(= 個別リターン − 東証プライム平均 − 同日ウィンドウセットの横断平均) を予測する
モデルを作る。

### 主要指標: `val_ic_spearman` (Spearman Information Coefficient)
`prepare.py` の `evaluate_all_metrics()` が返す `ic_spearman` を **最優先の評価指標**
とする。IC は予測 20 日累積リターンと実現 20 日累積残差リターンの順位相関で、
次の性質を持つ:

- サブセット選択効果に引っ張られない (Sharpe の弱点)
- 大きな n (val=51000) でのサンプリング誤差は ≈ 0.0044、IC ≥ 0.02 で 4σ 超
- 0 なら「ランキング能力ゼロ」、0.03 超なら業界でも「使える」水準

### 補助指標
- `val_sharpe`: 選択的ロング戦略 (pred > 0) の年率 Sharpe
- `val_always_long`: 全ウィンドウ毎日ロングした場合の Sharpe。**二重中心化のため
  構造的にゼロ** (絶対値 < 0.05 になっているはず)。もし離れていたらパイプラインに
  バグがあるサイン
- `val_dir_acc_20d`: 20 日累積の符号的中率。50% がベースライン

### ゴール (停止条件)
`train.py` を繰り返し編集・実行し、以下を **3 回連続** で満たしたら正常終了:

- `val_ic_spearman ≥ 0.02`
- `val_sharpe > val_always_long + 0.10` (モデル単体で常時ロングを明確に上回る)

達成したら改善ループを終了し、`artifacts/model.pt` を最良モデルとして残す。
15 イテレーションで打ち切り。

## 編集可能 / 不可
- **編集可 (train.py のみ)**: モデル構造、損失関数、オプティマイザ、学習率、バッチ
  サイズ、エポック数、特徴量の選び方 (FEATURE_COLS の部分集合を使う等)、
  シグナル閾値 (SIGNAL_THRESHOLD)、スケジューラ等。
- **編集不可**:
  - `prepare.py` — データパイプライン・特徴量の定義・`evaluate_sharpe` は固定
    (メトリクスの一貫性のため)。
  - `universe.py` — 銘柄ユニバース。
  - `requirements.txt` — 新規パッケージ追加禁止。
- **pip install は禁止**。torch / numpy / pandas / scikit-learn / joblib / yfinance だけで戦うこと。

## ループの手順 (各イテレーション)
1. `results.tsv` を Read して履歴と現状の best_ic、現在の連続達成数を把握する。
2. `train.py` を Read して現在の実装を確認する。
3. 改善アイデアを **1 つ** 選ぶ (同時に複数変えない)。下の Idea Bucket 参照。
4. Edit で `train.py` を変更し、`NOTES` 変数に変更内容を 1 行で書く。
5. Bash: `/c/Users/matsu/anaconda3/python.exe train.py > run.log 2>&1`
   - timeout 1200 秒 (20 分) を目安に実行。超えたら強制停止して discard。
6. 結果取り出し:
   `grep "^val_ic_spearman:\|^val_sharpe:\|^val_always_long:\|^val_dir_acc_20d:" run.log`
   - grep が空ならクラッシュ。`tail -n 60 run.log` でスタックトレース確認 → 自明な
     修正ならリトライ、そうでなければ revert して次のアイデアへ。
7. **採否判断** (主指標は IC):
   - `val_ic_spearman > best_ic` なら keep
   - そうでなければ Edit で直前の変更を巻き戻す (revert)
   - クラッシュも revert 扱い
8. **連続達成カウンタ**:
   - keep した run が `val_ic_spearman ≥ 0.02 AND val_sharpe − val_always_long ≥ 0.10`
     の両方を満たすなら counter += 1、そうでなければ counter = 0
   - counter が 3 に到達したら **ループ終了**
9. 安全弁: **15 イテレーション** までで終了。達成できなかった場合は、
   その時点で best の状態で終了してユーザーに報告する。

## Idea Bucket (順不同、適宜組み合わせ可)
- 損失関数の変更: MSE → Huber / SmoothL1 / MSE + sign-agreement 補助項
- 出力を 20 次元ベクトル → 20 日累積リターン 1 スカラーに縮約
- モデル: LSTM → GRU / 2 層化 / hidden 32〜128 の範囲で調整
- 軽量 Transformer (nn.TransformerEncoder, 1〜2 層, d_model=64)
- LSTM 出力に Attention pooling (学習可能な attention weight で時系列を集約)
- 学習率スケジュール: Cosine / Linear warmup / ReduceLROnPlateau
- Dropout / LayerNorm / weight decay の調整
- シグナル閾値 SIGNAL_THRESHOLD を 0 → 0.01 などに上げ、確信度の高いポジションのみ取る
- 特徴量を部分集合にする (ノイズの多い特徴を外す)
- 入力を差分/正規化しなおす (最後の 1 日を 0 基準にシフト等)
- 目標リターンを標準化してから学習 (損失スケール調整)
- 予測の符号に基づくランキング方式 (上位 N% を long など) を SIGNAL_THRESHOLD で近似
- エポック数を増やす / Early stopping を追加

**やりすぎ注意**: 1 回の変更は焦点を絞る。大規模書き換えは挙動の原因を追えなくなる。

## 出力契約 (train.py が壊してはいけないもの)
- stdout に `---` の区切り線の後、少なくとも以下 4 行を出す:
  - `val_ic_spearman:  +X.XXXX`
  - `val_sharpe:       +X.XXXX`
  - `val_always_long:  +X.XXXX`
  - `val_dir_acc_20d:  XX.XX%`
- `results.tsv` に 1 行 append する (ヘッダは自動生成される、9 列スキーマ)
- `artifacts/model.pt` に最良エポック時点の state_dict と config を torch.save する。
  最良エポックは **IC 最大** で選ぶこと (Sharpe はサブセット効果でノイジー)

この 3 つが守られていれば、ループ側が結果を拾えるので、内部実装は自由に変えてよい。

## 停止条件 (再掲)
- 以下 2 条件を両方 **3 回連続** で満たす → 正常終了
  1. `val_ic_spearman ≥ 0.02`
  2. `val_sharpe − val_always_long ≥ 0.10`
- **15 イテレーション** で打ち切り → 時点 best で終了

## 終了後のアクション
ループ終了時、以下のことを報告する:
- best_sharpe とそれを出した iteration
- results.tsv の要約 (keep / revert / crash の数)
- `artifacts/model.pt` の最新コンフィグ
- Streamlit アプリ (`app.py`) で試すためのコマンド
