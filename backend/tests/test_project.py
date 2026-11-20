import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from ml_modelcontrol import load_spec

from ml_createarchitecture.app import create_app

# 全結合だけの小さなモデルの定義
SPEC = {
    "format_version": 1,
    "name": "head",
    "inputs": {"x": {"shape": [4, 3]}},
    "nodes": {
        "fc": {
            "unit": "nn.Linear",
            "params": {"in_features": 3, "out_features": 2},
            "inputs": "x"
        }
    },
    "outputs": {"y": "fc"}
}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """
    トークンを Cookie に持つテスト用のクライアントを作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    client: TestClient
        API を呼ぶクライアント
    """
    # 一時ディレクトリを置き場にして, トークンの URL を開いた状態に設定
    app = create_app(
        data_dir=tmp_path / "home", token="secret",
        allowed_hosts=["testserver"]
    )
    client = TestClient(app)
    client.get("/api/auth?token=secret", follow_redirects=False)

    return client


def stub_inputs(path: Any) -> dict[str, Any]:
    """
    ML-EDA の model_inputs の代わりに, 決まった結果を返す関数

    Parameters
    ----------
    path: Any
        データセットのディレクトリ (dataset.yaml の中の版で振る舞いを変える)

    Returns
    ----------
    result: dict[str, Any]
        model_inputs と同じ形の結果
    """
    # 版 2 と説明の無い場合は ML-EDA と同じ例外
    description = Path(path) / "dataset.yaml"
    if not description.is_file():
        raise FileNotFoundError(str(description))
    version = yaml.safe_load(description.read_text())["format_version"]
    if version != 1:
        raise ValueError(f"format_version {version} は読めません")
    result = {
        "inputs": [{"name": "x", "shape": [None, 7], "dtype": "float32"}],
        "task": "classification", "target": "label",
        "classes": ["a", "b", "c"], "outputs": 3
    }

    return result


def write_dataset(path: Path, version: int) -> Path:
    """
    版だけを書いたデータセットの説明を作る関数

    Parameters
    ----------
    path: Path
        データセットのディレクトリ
    version: int
        説明の版

    Returns
    ----------
    path: Path
        データセットのディレクトリ
    """
    path.mkdir(parents=True)
    (path / "dataset.yaml").write_text(f"format_version: {version}\n")

    return path


def test_dataset_inputs_stub(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    データセットの入力と出す値の数を ML-EDA から読み, バッチの次元を例の
    大きさにすることを, 代わりの ML-EDA で確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え

    Returns
    ----------
    None
    """
    # 代わりの ML-EDA
    monkeypatch.setitem(
        sys.modules, "ml_eda", SimpleNamespace(model_inputs=stub_inputs)
    )
    dataset = write_dataset(tmp_path / "ds", 1)
    response = client.post(
        "/api/dataset/inputs", json={"path": str(dataset)}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputs"] == [
        {"name": "x", "shape": [4, 7], "dtype": "float32"}
    ]
    assert body["outputs"] == 3 and body["task"] == "classification"
    assert body["classes"] == ["a", "b", "c"]
    assert body["path"] == str(dataset.resolve())

    # 版 2 は ML-EDA の誤りの文で 422, 説明が無い場合は 404
    other = write_dataset(tmp_path / "ds2", 2)
    response = client.post("/api/dataset/inputs", json={"path": str(other)})
    assert response.status_code == 422
    assert "format_version 2" in response.json()["detail"]
    empty = tmp_path / "empty"
    empty.mkdir()
    response = client.post("/api/dataset/inputs", json={"path": str(empty)})
    assert response.status_code == 404
    assert "dataset.yaml" in response.json()["detail"]

    # 無いディレクトリは 404, 相対のパスは 422
    response = client.post(
        "/api/dataset/inputs", json={"path": str(tmp_path / "none")}
    )
    assert response.status_code == 404
    response = client.post("/api/dataset/inputs", json={"path": "ds"})
    assert response.status_code == 422


def test_dataset_inputs_missing_library(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ML-EDA が無い場合と古い場合に, 入れ方を 501 で返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え

    Returns
    ----------
    None
    """
    # 入っていない場合 (import できない)
    dataset = write_dataset(tmp_path / "ds", 1)
    monkeypatch.setitem(sys.modules, "ml_eda", None)
    response = client.post(
        "/api/dataset/inputs", json={"path": str(dataset)}
    )
    assert response.status_code == 501
    assert "make setup" in response.json()["detail"]

    # model_inputs の無い古い版
    monkeypatch.setitem(sys.modules, "ml_eda", SimpleNamespace())
    response = client.post(
        "/api/dataset/inputs", json={"path": str(dataset)}
    )
    assert response.status_code == 501
    assert "model_inputs" in response.json()["detail"]


def test_dataset_inputs_real(client: TestClient, tmp_path: Path) -> None:
    """
    本物の ML-EDA で書き出した表のデータセットから, 入力と出す値の数を
    読むことを確かめる関数

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
    # model_inputs のある ML-EDA だけ
    ml_eda = pytest.importorskip("ml_eda")
    if not hasattr(ml_eda, "model_inputs"):
        pytest.skip("ML-EDA に model_inputs がありません")

    # 3 クラスの分類の表を書き出し
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.normal(size=60), "b": rng.normal(size=60),
        "c": rng.normal(size=60),
        "label": np.array(["x", "y", "z"] * 20)
    })
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "target": "label", "task": "classification", "steps": [],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2}
    }
    ml_eda.export_dataset(
        ml_eda.load_data(source), recipe, tmp_path / "ds"
    )

    # 入力は学習に使う列の数, 出す値の数はクラスの数
    response = client.post(
        "/api/dataset/inputs", json={"path": str(tmp_path / "ds")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["shape"] for item in body["inputs"]] == [[4, 3]]
    assert body["outputs"] == 3 and body["target"] == "label"


def make_project_dir(path: Path) -> Path:
    """
    問題の設定の無い, 学習のプロジェクトの形のディレクトリを作る関数

    Parameters
    ----------
    path: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    path: Path
        プロジェクトのディレクトリ
    """
    config = path / "configs" / "default" / "train.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model:\n  spec: models/model.yaml\n")

    return path


def test_project_dir_save(client: TestClient, tmp_path: Path) -> None:
    """
    記録の無いプロジェクトでは骨組みの既定の場所に保存し, 既にある場合は
    上書きを確かめることを確かめる関数

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
    # 保存先は models/model.yaml (まだ無い)
    project = make_project_dir(tmp_path / "proj")
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 200, response.text
    target = response.json()
    expected = project.resolve() / "models" / "model.yaml"
    assert target["path"] == str(expected)
    assert target["source"] == "project" and target["exists"] is False

    # 保存 (models/ を作る) すると定義として読める
    body = {"path": str(project), "spec": SPEC}
    response = client.put("/api/project/model", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(expected)
    assert load_spec(expected)["nodes"]["fc"]["unit"] == "nn.Linear"
    assert not list(expected.parent.glob("*.tmp"))

    # 既にある場合は上書きの許しが要る (409)
    response = client.put("/api/project/model", json=body)
    assert response.status_code == 409
    changed = {**SPEC, "outputs": {"z": "fc"}}
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "spec": changed, "overwrite": True}
    )
    assert response.status_code == 200
    assert load_spec(expected)["outputs"] == {"z": "fc"}
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.json()["exists"] is True

    # 定義として読めないものは書かない
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "spec": {"format_version": 99},
              "overwrite": True}
    )
    assert response.status_code == 422
    assert load_spec(expected)["outputs"] == {"z": "fc"}


def test_project_target_errors(client: TestClient, tmp_path: Path) -> None:
    """
    プロジェクトでないディレクトリ, 無い場所, ファイルでない保存先を
    断ることを確かめる関数

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
    # config の無いディレクトリは 422, 無い場所は 404
    plain = tmp_path / "plain"
    plain.mkdir()
    response = client.post("/api/project/target", json={"path": str(plain)})
    assert response.status_code == 422
    assert "configs/default/train.yaml" in response.json()["detail"]
    response = client.post(
        "/api/project/target", json={"path": str(tmp_path / "none")}
    )
    assert response.status_code == 404

    # 保存先にディレクトリがある場合は 422
    project = make_project_dir(tmp_path / "proj")
    (project / "models" / "model.yaml").mkdir(parents=True)
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "spec": SPEC, "overwrite": True}
    )
    assert response.status_code == 422


class StubQuest:
    def __init__(self, record: dict[str, Any]):
        # handoff が返す値を保存
        self.record = record

    def load_problem(self, path: Any) -> dict[str, Any]:
        """
        問題の設定の YAML をそのまま読む関数

        Parameters
        ----------
        path: Any
            問題の設定のファイル

        Returns
        ----------
        problem: dict[str, Any]
            読んだ値
        """
        problem: dict[str, Any] = yaml.safe_load(Path(path).read_text())

        return problem

    def handoff(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        決まった受け渡しの値を返す関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定 (使わない)

        Returns
        ----------
        record: dict[str, Any]
            保存した値
        """
        record = self.record

        return record


def test_problem_record(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    問題の設定の記録 (model.project_model_spec) の保存先に書き, プロジェ
    クトの外と記録の無い問題の設定を断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え

    Returns
    ----------
    None
    """
    # 記録の保存先 (config の既定と違う名前) とデータセットの場所を返す
    project = make_project_dir(tmp_path / "proj").resolve()
    problem = tmp_path / "store" / "sales.yaml"
    problem.parent.mkdir()
    problem.write_text("name: sales\n")
    spec_path = project / "models" / "custom.yaml"
    record = {
        "model": {"project_model_spec": str(spec_path)},
        "data": {"project_dataset": str(project / "outputs" / "dataset")},
        "project": {"path": str(project)}
    }
    monkeypatch.setitem(sys.modules, "ml_quest", StubQuest(record))
    response = client.post(
        "/api/project/target", json={"path": str(problem)}
    )
    assert response.status_code == 200, response.text
    target = response.json()
    assert target["path"] == str(spec_path) and target["source"] == "problem"
    assert target["dataset"] == str(project / "outputs" / "dataset")
    assert target["problem"] == "sales"
    response = client.put(
        "/api/project/model", json={"path": str(problem), "spec": SPEC}
    )
    assert response.status_code == 200 and spec_path.is_file()

    # プロジェクトの中の problem.yaml の記録も同じ保存先
    (project / "problem.yaml").write_text("name: sales\n")
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.json()["path"] == str(spec_path)

    # プロジェクトの外を指す記録は 422
    record["model"]["project_model_spec"] = str(tmp_path / "other.yaml")
    response = client.post(
        "/api/project/target", json={"path": str(problem)}
    )
    assert response.status_code == 422
    assert "外" in response.json()["detail"]

    # 記録の無い問題の設定のファイルは 422. プロジェクトの中の記録の無い
    # problem.yaml は骨組みの既定
    record["model"] = {}
    response = client.post(
        "/api/project/target", json={"path": str(problem)}
    )
    assert response.status_code == 422
    assert "ML-Pathfinder" in response.json()["detail"]
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.json()["path"] == str(project / "models" / "model.yaml")


def test_problem_real(client: TestClient, tmp_path: Path) -> None:
    """
    本物の ML-Quest が書き出した骨組みのプロジェクトに保存することを
    確かめる関数

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
    # 例の問題の設定から骨組みを書き出し
    ml_quest = pytest.importorskip("ml_quest")
    problem = ml_quest.example_problem(next(iter(ml_quest.EXAMPLES)))
    project = tmp_path / "proj"
    ml_quest.make_project(problem, project)

    # 保存先は config の model.spec (記録があればそれも同じ場所)
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 200, response.text
    expected = project.resolve() / "models" / "model.yaml"
    assert response.json()["path"] == str(expected)
    response = client.put(
        "/api/project/model", json={"path": str(project), "spec": SPEC}
    )
    assert response.status_code == 200 and expected.is_file()
