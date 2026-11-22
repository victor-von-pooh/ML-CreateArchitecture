"""
画面の通しのテスト用に, 書き出したデータセット, 学習のプロジェクトと,
ML-Pathfinder の置き場の問題の設定を作るスクリプト (ML-EDA と ML-Quest で
作る. 追加の依存 project が要る)
"""

import hashlib
import sys
from pathlib import Path

import ml_eda
import ml_quest
import numpy as np
import pandas as pd
import yaml


def make_dataset(path: Path) -> None:
    """
    3 列の 3 クラスの分類の表を, データセットとして書き出す関数

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 決まった乱数の表を CSV にして, ML-EDA で読んで書き出し
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.normal(size=60), "b": rng.normal(size=60),
        "c": rng.normal(size=60),
        "label": np.array(["x", "y", "z"] * 20)
    })
    source = path.parent / "table.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "target": "label", "task": "classification", "steps": [],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2}
    }
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, path)


def make_masks(path: Path) -> None:
    """
    灰色の画像と 3 つの値の領域の, ファイルの組のデータセット (版 3) の
    説明を書く関数 (model_inputs は説明だけを読むので分割のファイルは
    書かない)

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 大きさの混ざる画像 (高さと幅は決まらない) と, 背景 0 の名前の無い領域
    path.mkdir(parents=True)
    info = {
        "format_version": 3, "role": "predict",
        "target": {"kind": "mask", "entries": ["mask"]},
        "entries": {
            "image": {"kind": "image", "role": "input",
                      "shape": [None, None, 1], "dtype": "uint8",
                      "steps": ["grayscale"]},
            "mask": {"kind": "mask", "role": "target",
                     "shape": [None, None], "dtype": "uint8",
                     "values": [0, 1, 2],
                     "classes": [{"value": 1, "name": "circle"},
                                 {"value": 2, "name": "square"}]},
        },
    }
    (path / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True), encoding="utf-8"
    )


def make_project(path: Path) -> None:
    """
    ML-Quest の例の問題の設定から, 学習のプロジェクトの骨組みを書き出す
    関数

    Parameters
    ----------
    path: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 最初の例 (列の予測) の骨組み
    problem = ml_quest.example_problem(next(iter(ml_quest.EXAMPLES)))
    ml_quest.make_project(problem, path)


def make_store(path: Path, project: Path, dataset: Path) -> None:
    """
    ML-Pathfinder の置き場に, プロジェクトを書き出した問題の設定と, まだ
    書き出していない問題の設定を保存する関数

    Parameters
    ----------
    path: Path
        ML-Pathfinder の置き場
    project: Path
        書き出したプロジェクトのディレクトリ
    dataset: Path
        記録に書くデータセットのディレクトリ

    Returns
    ----------
    None
    """
    # 書き出した方は, 記録のデータセットを作ったデータセットにする (入力と
    # 出力を作れるように)
    problem = ml_quest.example_problem(next(iter(ml_quest.EXAMPLES)))
    recorded = ml_quest.record_project(
        {**problem, "name": "e2e の売上"}, project
    )
    recorded["project"]["dataset"] = str(dataset.resolve())
    ml_quest.save_problem(recorded, path / "problems" / "recorded")
    draft = {**problem, "name": "e2e の下書き"}
    ml_quest.save_problem(draft, path / "problems" / "draft")

    # 書き出してあり, モデルの定義の置き場の記録だけが無いもの
    no_spec = ml_quest.record_project(
        {**problem, "name": "e2e の定義の記録なし"}, project
    )
    no_spec["project"]["model_spec"] = None
    ml_quest.save_problem(no_spec, path / "problems" / "no_spec")


def make_experiment(project: Path) -> None:
    """
    プロジェクトの実験の置き場に, 定義の写しを持つ ML-Found の形の実験を
    作る関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 実験の置き場は config の experiment.output_dir (無ければ outputs)
    config = yaml.safe_load(
        (project / "configs" / "default" / "train.yaml").read_text()
    )
    root = (config.get("experiment") or {}).get("output_dir") or "outputs"
    run_dir = project / root / "e2e_run" / "2026-11-20" / "10-00-00"
    run_dir.mkdir(parents=True)
    spec = {
        "format_version": 1,
        "name": "e2e_run",
        "inputs": {"x": {"shape": [4, 3]}},
        "nodes": {
            "e2e_fc": {
                "unit": "nn.Linear",
                "params": {"in_features": 3, "out_features": 2},
                "inputs": "x",
            }
        },
        "outputs": {"y": "e2e_fc"},
    }
    data = yaml.safe_dump(spec, sort_keys=False).encode()
    (run_dir / "model_spec.yaml").write_bytes(data)
    (run_dir / "config.yaml").write_text("model:\n  spec: models/x.yaml\n")
    record = {
        "path": "models/e2e_run.yaml", "status": "copied",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (run_dir / "source.yaml").write_text(
        yaml.safe_dump({"format_version": 1, "model_spec": record})
    )
    (run_dir / "run.json").write_text('{"status": "completed"}')


if __name__ == "__main__":
    work = Path(sys.argv[1])
    make_dataset(work / "dataset")
    make_masks(work / "masks")
    make_project(work / "project")
    make_experiment(work / "project")
    make_store(work / "pathfinder", work / "project", work / "dataset")
