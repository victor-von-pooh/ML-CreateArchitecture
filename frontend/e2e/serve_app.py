"""
画面の通しのテスト用に, 異常の検知の問題の設定の受け持ちに合う雛形の名前を
ML-Quest の受け渡しに足して, アプリを起動するスクリプト (ML-Quest が異常の
検知の受け持ちを書けるまでの代わり. 追加の依存 project が要る)
"""

import importlib
from typing import Any, Optional

from ml_createarchitecture.cli import main

# 代わりに雛形の名前を足す問題の設定の名前 (make_fixtures.py と同じ)
ANOMALY_NAME = "e2e の異常の検知"

# 異常の検知の自己符号化器の雛形 (ML-ModelControl の同梱の例)
ANOMALY_EXAMPLE = "tabular_autoencoder"


def stand_in_example() -> None:
    """
    ML-Quest の受け渡しの雛形の名前を, 異常の検知の問題の設定では
    ANOMALY_EXAMPLE にする関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # handoff が呼ぶ model_example を, 名前で見分けて差し替え
    module = importlib.import_module("ml_quest.handoff")
    original = module.model_example

    def model_example(problem: dict[str, Any]) -> Optional[str]:
        """
        異常の検知の問題の設定だけ雛形の名前を返し, 他は ML-Quest に任せる
        関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定

        Returns
        ----------
        example: Optional[str]
            同梱の例の名前 (合う例が無ければ None)
        """
        # 名前で見分け, 他は元の決め方
        if problem.get("name") == ANOMALY_NAME:
            example: Optional[str] = ANOMALY_EXAMPLE
            return example
        example = original(problem)

        return example

    module.model_example = model_example


if __name__ == "__main__":
    stand_in_example()
    main()
