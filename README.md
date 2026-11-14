# ML-CreateArchitecture

[ML-ModelControl](https://github.com/victor-von-pooh/ML-ModelControl) の部品を GUI で並べてつなぎ, PyTorch のモデルを組み立てるローカルの Web アプリ.

部品 (`nn.Linear` のような層, 足し算や連結のような演算, PyTorch Geometric の層, Hugging Face のモデル) を一覧から選んで置き, 線でつなぎ, 引数をフォームで決める. 形とパラメータ数を確かめながら, モデルの定義 (YAML) を作って保存する.

設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照. 開発中のため, 導入と起動の手順は実装に合わせて追記する.

## ライセンス

MIT
