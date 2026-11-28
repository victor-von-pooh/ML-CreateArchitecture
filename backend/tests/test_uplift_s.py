from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_project import client, uplift_problem, write_uplift_dataset

from ml_createarchitecture.project import output_size

# 他のモジュールの fixture を, このモジュールのテストでも使う
__all__ = ["client"]


def s_learner_problem(ml_eda: Any, ml_quest: Any, root: Path) -> Path:
    """
    効果の見積もりで S-learner を選んだ問題の設定を, 骨組みと記録の
    データセット付きで保存する関数

    Parameters
    ----------
    ml_eda: Any
        ML-EDA のモジュール
    ml_quest: Any
        ML-Quest のモジュール
    root: Path
        プロジェクト, データセットと問題の設定を置く場所

    Returns
    ----------
    place: Path
        保存した問題の設定のファイル
    """
    # 問い uplift.learner で s_learner を選んだ問題の設定
    problem = ml_quest.answer(
        uplift_problem(ml_quest), "ml.parts.0.uplift.learner", "s_learner"
    )
    project = root / "project"
    ml_quest.make_project(problem, project)
    recorded = ml_quest.record_project(problem, project)
    recorded["project"]["dataset"] = str(write_uplift_dataset(ml_eda, root))
    place = root / "problem.yaml"
    ml_quest.save_problem(recorded, place)

    return place


def test_project_save_contract_s_learner(
    client: TestClient, tmp_path: Path
) -> None:
    """
    S-learner の問題の設定では, データセットの入力に約束の入力の offset
    (処置の列の 1) が付き, それを足した幅で書き換えた雛形 tabular_uplift_s
    が骨組みの約束の確かめを通り, 足さない幅の定義は断ることを確かめる
    関数 (ML-EDA と ML-Quest が無ければ飛ばす)

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 受け渡しの雛形と, データセットの入力の足す数
    ml_eda = pytest.importorskip("ml_eda")
    ml_quest = pytest.importorskip("ml_quest")
    if "s_learner" not in str(ml_quest.UPLIFT_LEARNERS):
        pytest.skip("ML-Quest に S-learner がありません")
    place = s_learner_problem(ml_eda, ml_quest, tmp_path)
    target = client.post(
        "/api/project/target", json={"path": str(place)}
    ).json()
    assert target["example"] == "tabular_uplift_s"
    info = client.post(
        "/api/dataset/inputs",
        json={"path": target["dataset"], "problem": str(place)}
    ).json()
    assert info["offsets"] == {"x": 1}
    columns = info["inputs"][0]["shape"][-1]
    assert info["uplift"]["columns"] == columns == 5

    # 足した幅 (学習に使う列の数 + 1) で書き換えた雛形は確かめを通る
    template = client.get("/api/examples/tabular_uplift_s").json()["spec"]
    adapt = {"spec": template, "name": "tabular_uplift_s"}
    response = client.post(
        "/api/templates/adapt",
        json={**adapt, "values": {"features": columns + 1}}
    )
    assert response.status_code == 200, response.text
    spec = response.json()["spec"]
    assert spec["inputs"]["x"]["shape"][-1] == columns + 1
    response = client.put(
        "/api/project/model", json={"path": str(place), "spec": spec}
    )
    assert response.status_code == 200, response.text
    assert response.json()["notes"] == []

    # 足さない幅 (学習に使う列の数) は, 入力 x を直す場所付きの 409
    response = client.post(
        "/api/templates/adapt", json={**adapt, "values": {"features": columns}}
    )
    narrow = response.json()["spec"]
    response = client.put("/api/project/model", json={
        "path": str(place), "spec": narrow, "overwrite": True
    })
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert [item["id"] for item in detail["findings"]] == ["inputs.x"]
    assert detail["widths"] == {"features": columns}


def test_dataset_inputs_t_learner_offsets(
    client: TestClient, tmp_path: Path
) -> None:
    """
    S-learner を選んでいない (T-learner の) 問題の設定では, データセットの
    入力に足す数が無いことを確かめる関数 (ML-EDA と ML-Quest が無ければ
    飛ばす)

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 問い uplift.learner に答えていない問題の設定
    ml_eda = pytest.importorskip("ml_eda")
    ml_quest = pytest.importorskip("ml_quest")
    dataset = write_uplift_dataset(ml_eda, tmp_path)
    place = tmp_path / "problem.yaml"
    ml_quest.save_problem(uplift_problem(ml_quest), place)
    response = client.post(
        "/api/dataset/inputs",
        json={"path": str(dataset), "problem": str(place)}
    )
    assert response.status_code == 200, response.text
    assert response.json()["offsets"] == {}


def test_output_size() -> None:
    """
    約束の出力の大きさの数が, ML-Quest の check_model_contract と同じく,
    鍵の数に足す数を足し, 鍵の数が無ければ決まった数になることを確かめる
    関数

    Returns
    ----------
    None
    """
    # 鍵の数と足す数, 足す数の無い項目, 鍵の数の無い項目
    widths = {"features": 5}
    assert output_size({"width": "features", "offset": 1}, widths) == 6
    assert output_size({"width": "features", "size": 2}, widths) == 5
    assert output_size({"width": "levels", "size": 3}, widths) == 3
    assert output_size({"width": None, "size": 1}, widths) == 1
    assert output_size({"width": "levels"}, widths) is None
