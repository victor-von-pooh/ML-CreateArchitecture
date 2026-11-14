# ML-CreateArchitecture

[ML-ModelControl](https://github.com/victor-von-pooh/ML-ModelControl) の部品を GUI で並べてつなぎ, PyTorch のモデルを組み立てるローカルの Web アプリ.

部品 (`nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric の層, Hugging Face のモデル) を一覧から選んで置き, 線でつなぎ, 引数をフォームで決める. 形とパラメータ数を確かめながら, モデルの定義 (YAML) を作って保存する.

設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照.

## 必要なもの

- Python 3.10 以上と [uv](https://docs.astral.sh/uv/). ML-ModelControl の依存の PyTorch も入るので, 最初の `make setup` は時間がかかる
- Node.js 24 以上と [pnpm](https://pnpm.io/)

## 導入と起動

```bash
git clone --recursive https://github.com/victor-von-pooh/ML-CreateArchitecture.git
cd ML-CreateArchitecture
make setup   # submodule, Python と Node.js の依存
make serve   # 画面をビルドして http://127.0.0.1:8766 で起動 (端末を占有するので, 止めるまで使い続ける)
```

起動すると, 起動ごとのトークンが付いた URL (`http://127.0.0.1:8766/api/auth?token=...`) が表示されるので, それを開く. 毎回同じトークンにする場合は, 環境変数 `ML_CREATEARCHITECTURE_TOKEN` で指定する. ポートは `make serve PORT=8800` のように変えられる.

外部のライブラリ (PyTorch Geometric, Hugging Face, torchvision, timm, torchaudio, einops, Kornia, MONAI, segmentation_models_pytorch) の部品を使う場合は, `make setup` に `EXTRAS` を付けてバックエンドに入れる (入れていないライブラリの部品は, 一覧に「使えない」と理由が出る). 全て入れると数 GB になり時間がかかる. `EXTRAS` を付けずに `make setup` をやり直すと, 入れたライブラリは外れる.

```bash
make setup EXTRAS=all                       # 全て
make setup EXTRAS="geometric huggingface"   # 一部だけ
```

torchvision と timm のモデル (`tv.Model`, `timm.Model`) は, 右の詳細の名前の欄に打つと候補が出る.

## 使い方

1. 左の一覧から部品を選ぶ (名前と説明で検索できる). 押すとキャンバスに置かれる
2. 口 (ノードの左右の点) から口へドラッグして線を引く. 左が入力の口 (`?` は省略できる口, `…` は複数つなげる口), 右が出力の口. 線とノードは選んで Delete で消せる. 一覧に無い引数を受け取る部品 (einops の式の大きさ `h=2` など) は, 右の「一覧に無い引数」で名前を書いて加える. 保存した定義は「開く」の一覧のごみ箱で消せる. ノードは右の「複製」か Ctrl+D で引数ごと複製でき, Ctrl+C と Ctrl+V でも写して貼れる (つなぎ先は写さない). 上の「整列」で, つながりの順に左から右へ並べ直せる (位置を書いていない定義を開いた場合も, 同じ並べ方で置く). 間違えた操作は Ctrl+Z (Mac は Cmd+Z) か上の「元に戻す」で戻せる (Ctrl+Shift+Z でやり直す. 定義を開くか新規にすると履歴は消える). 線を引く代わりに, ノードを選んで右の「入力のつなぎ先」から選んでもつなげる (出力も同じ. キーボードだけでも操作できる)
3. ノードを選ぶと, 右に引数のフォーム (説明と範囲付き. `*` は必須) が出る. 推定した出力の形とパラメータ数, 誤りもここに出る
4. 上の「入力」と「出力」でモデルの入力と出力を足す. 入力は形 (`[4, 16]` のように, バッチの次元も含む) と型 (単語の番号や `edge_index` は `int64`) を決める
5. 名前を付けて「保存」. 「開く」で保存した定義か, 例 (全結合, 残差, CNN, LSTM, Transformer, GCN, BERT の分類) を開き, 「YAML」で保存する形, 「Python」で同じ計算をするコードを見られる

保存した定義は `~/.ml-createarchitecture/models/` に YAML で置かれる (環境変数 `ML_CREATEARCHITECTURE_HOME` で変えられる). ML-ModelControl の `build_model` でそのまま `nn.Module` にできる. ML-Found で学習する手順は ML-ModelControl の README の「ML-Found で学習する」にある (定義のファイルの置き場をプロジェクトの config に書き, 学習スクリプトで組み立てる).

## 開発

```bash
make dev-backend    # API (変更を自動で読み直す)
make dev-frontend   # 画面 (別の端末で. http://localhost:5173)
make check          # lint, 型検査, テスト
make api            # API を変えたら, 画面の型を作り直す
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
