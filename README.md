# ML-CreateArchitecture

[ML-ModelControl](https://github.com/victor-von-pooh/ML-ModelControl) の部品を GUI で並べてつなぎ, PyTorch のモデルを組み立てるローカルの Web アプリ.

部品 (`nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric の層, Hugging Face, torchvision, timm などのモデル) を一覧から選んで置き, 線でつなぎ, 引数をフォームで決める. 形とパラメータ数を確かめながら, モデルの定義 (YAML) を作って保存する.

設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照.

## 必要なもの

- [uv](https://docs.astral.sh/uv/) (Python は uv が用意する. 3.10 以上). ML-ModelControl の依存の PyTorch も入るので, 最初の `make setup` は時間がかかる
- Node.js 24 以上と [pnpm](https://pnpm.io/) 11 以上 (`corepack enable` で `package.json` の版が入る)

## 導入と起動

```bash
git clone --recursive https://github.com/victor-von-pooh/ML-CreateArchitecture.git
cd ML-CreateArchitecture
make setup   # submodule, Python と Node.js の依存
make serve   # 画面をビルドして http://127.0.0.1:8766 で起動 (使う間は端末を開いたままにする. Ctrl+C で止める)
```

起動すると, 画面のビルドのログの後に `ML-CreateArchitecture を開く: http://127.0.0.1:8766/api/auth?token=...` の行が出るので, その URL をブラウザで開く (トークンは起動ごとに変わる). 毎回同じトークンにする場合は, 環境変数 `ML_CREATEARCHITECTURE_TOKEN` で指定する. ポートは `make serve PORT=8800` のように変えられる.

外部のライブラリ (PyTorch Geometric, Hugging Face, torchvision, timm, torchaudio, einops, Kornia, MONAI, segmentation_models_pytorch) の部品を使う場合は, `make setup` に `EXTRAS` を付けてバックエンドに入れる (入れていないライブラリの部品は, 一覧に「使えない」と理由が出る). 全て入れると数 GB になり時間がかかる. `EXTRAS` を付けずに `make setup` をやり直すと, 入れたライブラリは外れる.

```bash
make setup EXTRAS=all                       # 全て
make setup EXTRAS="torchvision timm"        # 一部だけ
```

`EXTRAS` に書ける名前は `geometric` (PyTorch Geometric), `huggingface`, `torchvision`, `timm`, `torchaudio`, `einops`, `kornia`, `monai`, `smp` (segmentation_models_pytorch), `all`.

torchvision と timm のモデル (`tv.Model`, `timm.Model`) の名前は, 右の詳細の引数 `name` (timm は `model_name`) の欄に打つと候補が出る.

## 使い方

### 最初のモデル (全結合の層 1 つ)

新規の画面には, 入力 `x` (形 `[4, 16]`) が最初からある.

1. キャンバスの入力 `x` を押して選ぶ
2. `/` を押して左の一覧の検索欄に移り (欄が畳まれていれば開く), `nn.Linear` と打って Enter を押す (一覧の `nn.Linear` を押してもよい). `x` の右に `linear1` が置かれ, `x` から `linear1` の口 `input` へ線が引かれる (何も選ばずに置いた場合は, `x` の右の点から `linear1` の左の点へドラッグして線を引く)
3. 右の詳細で `in_features` に `16`, `out_features` に `4` を入れる (`*` は必須の引数. 説明と範囲が欄の下に出る)
4. 上の「出力」(アイコンだけの場合は右向きの矢印) を押す. つながっていないノードが `linear1` だけなので, 出力は `linear1` につながる
5. 上の欄が「パラメータ数 68」「誤り 0 件」になり, ノードに出力の形 `[4, 4]` が出る
6. 上の名前の欄に名前 (英数字, `_`, `-`) を書いて「保存」(Ctrl+S でもよい). 少しの間「<名前> に保存しました」と出る

「開く」から例 (全結合, 残差, CNN, LSTM, Transformer, GCN, BERT の分類, 部分モデルを重ねたもの) を開いて, 組み立て方を見るのもよい.

### 画面

- **左**: 部品の一覧 (出どころごと. 名前と説明で検索). 使えない部品 (ライブラリが無いなど) は理由が出る. 「別の定義を部品にする」で加えた部分モデルは一番上の「部分モデル」に出る
- **中央**: キャンバス. 左の点が入力の口 (`?` は省略できる口, `…` は複数つなげる口), 右の点が出力の口. ノードに推定した出力の形が出る. 部分モデルの中身を編集している間は, 上に階層 (`本体 › sub.<名前>`) が出る
- **右**: 選んだものの詳細. 名前, 入る形と出る形, パラメータ数, 誤り, 入力のつなぎ先 (選択の欄. キーボードだけでもつなげる), 引数, 一覧に無い引数 (einops の式の大きさ `h=2` など). 何も選んでいないときは全体の情報と誤りの一覧
- **上**: 新規, 開く (保存した定義と例. ごみ箱で消せる), 保存, 元に戻す・やり直す, 整列, 入力, 出力, YAML (保存する形), Python (同じ計算をするコード. ML-ModelControl を入れていない環境でも動く. 外部のライブラリの部品はそのライブラリが要る), 明るさの切り替え. YAML と Python はコピーとファイルとしての保存ができる. 幅の狭い画面では一部がアイコンだけになる (マウスを重ねると名前が出る)

左右の欄は, キャンバスの左上と右上のボタンで畳める. 幅 1200 px 未満の画面では畳んだ状態で始まる.

モデルの入力の形 (`[4, 16]` のようにバッチの次元も含めて書く) と型 (単語の番号や `edge_index` は `int64`) は, 形の推定にだけ使う. 作ったモデルの入力の大きさを固定するものではない.

### 操作

キーの Ctrl は, Mac では Cmd.

| 操作 | 方法 |
| --- | --- |
| 部品を置く | 一覧の部品を押す (見えている範囲の真ん中に置く) か, キャンバスにドラッグする (落とした位置に置く) |
| 部品を探す | `/` で左の検索欄に移る. Enter で, 名前が合う使える部品を置き, 詳細の引数の最初の欄に移る |
| 線を引く | 出力の点から入力の点へドラッグ, または詳細の「入力のつなぎ先」で選ぶ. 入力かノードを選んだまま部品を押すか Enter で置くと, つなぎ方が 1 通りに決まる場合はそこから線を引く (ドラッグで置いた場合は引かない). 出力を加えると, つながっていないノードが 1 つだけならそれにつなぐ |
| 消す | 線かノードを選んで Delete か Backspace (または詳細の「ノードを消す」) |
| 元に戻す・やり直す | Ctrl+Z で戻し, Ctrl+Shift+Z (か Ctrl+Y) でやり直す. 引数の欄の中では, 打ちかけの文字がある間は文字の取り消し, 反映した後は定義を戻す. 定義を開くか新規にすると履歴は消える |
| 複製 | 詳細の「複製」か Ctrl+D. Ctrl+C と Ctrl+V でも写して貼れる (つなぎ先は写さない) |
| 並べ直す | 上の「整列」(つながりの順に左から右へ) |
| 保存 | 上の「保存」か Ctrl+S. 開いたものと違う名前で既にある定義に保存する場合は確かめる |
| 別の定義を部品にする | 左の「別の定義を部品にする」で保存した定義か例を選ぶと, 一覧の「部分モデル」に `sub.<名前>` の部品として出る (定義の入力が入力の口, 出力が出力の口). 定義の中に写すので, 元の定義を後で変えても変わらない |
| 部分モデルを写し直す | 元の定義を直した後に, 一覧の部分モデルの写し直しのボタンを押すと, 元の定義の今の中身に置き換わる |
| 部分モデルを消す | 一覧の部分モデルのごみ箱 (使っているノードが無いものだけ) |
| 部分モデルの中身を編集する | 部分モデルのノードをダブルクリックするか, 詳細の「部分モデルの中身を編集」. 変更は, その部分モデルを使う全てのノードに効く. Esc で 1 つ上の階層に戻る (上の階層の名前を押してもよい) |
| 部分モデルを 1 つのノードだけ変える | 部分モデルのノードの詳細の「このノードだけの部分モデルにする」(2 つ以上のノードが使う部分モデルの場合に出る. 写して別の部分モデルにし, 他のノードはそのまま) |

### 保存した定義を使う

保存した定義は `~/.ml-createarchitecture/models/<名前>.yaml` に置かれる (環境変数 `ML_CREATEARCHITECTURE_HOME` で変えられる). [ML-ModelControl](https://github.com/victor-von-pooh/ML-ModelControl) の `build_model` でそのまま `nn.Module` にできる. ML-ModelControl はこのアプリのバックエンドに入っているので, 試すだけなら `cd backend && uv run python` で使える (自分の環境では `pip install "ml-modelcontrol @ git+https://github.com/victor-von-pooh/ML-ModelControl.git"`).

```python
from pathlib import Path
from ml_modelcontrol import build_model

model = build_model(Path("~/.ml-createarchitecture/models/first_mlp.yaml").expanduser())
```

ML-Found で学習する場合は, ML-ModelControl の README の [ML-Found で学習する](https://github.com/victor-von-pooh/ML-ModelControl#ml-found-で学習する) を参照.

## 開発

```bash
make dev-backend    # API (変更を自動で読み直す. 開く URL を出す)
make dev-frontend   # 画面 (別の端末で. make dev-backend が出す http://localhost:5173/api/auth?token=... を開く)
make check          # lint, 型検査, テスト
make api            # API を変えたら, 画面の型を作り直す
cd frontend && pnpm exec playwright install chromium   # 通しのテストのブラウザ (最初の 1 回)
cd frontend && pnpm e2e   # 画面の通しのテスト (アプリを起動してブラウザで操作する)
```

| ディレクトリ | 内容 |
| --- | --- |
| `backend/` | API (Python, FastAPI). ML-ModelControl の確認, 形の推定, 書き出しを呼び, 定義のファイルを保存する |
| `frontend/` | 画面 (TypeScript, React, Vite, React Flow) |
| `ML-ModelControl/` | submodule |
| `docs/DESIGN.md` | 設計の方針と段階 |

## ライセンス

MIT
