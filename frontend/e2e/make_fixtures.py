"""
画面の通しのテスト用に, 書き出したデータセットと学習のプロジェクトを作る
スクリプト (ML-EDA と ML-Quest で作る. 追加の依存 project が要る)
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


if __name__ == "__main__":
    work = Path(sys.argv[1])
    make_dataset(work / "dataset")
    make_project(work / "project")
