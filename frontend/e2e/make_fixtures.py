"""
画面の通しのテスト用に, 書き出したデータセット, 学習のプロジェクトと,
ML-Pathfinder の置き場の問題の設定を作るスクリプト (ML-EDA と ML-Quest で
作る. 追加の依存 project が要る)
"""

import sys
from pathlib import Path

import ml_eda
import ml_quest
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    work = Path(sys.argv[1])
    make_dataset(work / "dataset")
    make_project(work / "project")
    make_store(work / "pathfinder", work / "project", work / "dataset")
