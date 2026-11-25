#!/usr/bin/env bash
# 画面の通しのテスト (Playwright) 用に, 作業用の置き場でアプリを起動する.
# 画面は pnpm build で作っておく
set -euo pipefail

cd "$(dirname "$0")"
WORK="$PWD/.work"
MODULES="$PWD/custom_modules"
FIXTURES="$PWD/make_fixtures.py"
PORT="${E2E_PORT:-8792}"

# 前回の結果を消して, 決まったトークンで起動. 学習済みの重みのライセンスなどで
# Hugging Face を読みに行かないように, ネットワークを使わない設定 (HF_HUB_OFFLINE)
rm -rf "$WORK"
mkdir -p "$WORK/home/models"
cd ../../backend

# 外部の重みの読み込みを確かめる小さな重みのファイル (定義の置き場に置く)
uv run python -c "
import sys, torch
torch.save({'model.fc.weight': torch.ones(4, 16),
            'model.head.weight': torch.zeros(3, 4)}, sys.argv[1])
" "$WORK/home/models/e2e_base.pt"
# データセットから入力と出力を作る, プロジェクトに保存するための, 書き出した
# データセット, 学習のプロジェクトと ML-Pathfinder の置き場
uv run python "$FIXTURES" "$WORK"
# 自作の部品は, 起動の設定 (ML_CREATEARCHITECTURE_MODULES) に書いた見本の
# モジュールだけを読み込む. 問題の設定の一覧は作業用の ML-Pathfinder の置き場
ML_PATHFINDER_HOME="$WORK/pathfinder" \
  ML_CREATEARCHITECTURE_HOME="$WORK/home" ML_CREATEARCHITECTURE_TOKEN=e2e-token \
  ML_CREATEARCHITECTURE_MODULES=e2e_lab_units PYTHONPATH="$MODULES" \
  HF_HUB_OFFLINE=1 exec uv run ml-createarchitecture --port "$PORT"
