import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient
from ml_modelcontrol import load_spec

from ml_createarchitecture import project as project_module
from ml_createarchitecture.app import create_app
from ml_createarchitecture.project import (
    ignores_case,
    make_parents,
    positive_int,
    same_file,
)

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
    # 知らない版と説明の無い場合は ML-EDA と同じ例外
    description = Path(path) / "dataset.yaml"
    if not description.is_file():
        raise FileNotFoundError(str(description))
    info = yaml.safe_load(description.read_text())
    version = info["format_version"]
    if version != 1:
        raise ValueError(f"format_version {version} は読めません")
    result = {
        "inputs": [{"name": "x", "shape": [None, 7], "dtype": "float32"}],
        "task": "classification", "target": "label",
        "classes": ["a", "b", "c"], "outputs": 3
    }
    # 説明に vocabulary があれば文の語の番号の入力 (語彙の大きさ付き)
    if "vocabulary" in info:
        result["inputs"] = [{"name": "x", "shape": [None, 16],
                             "dtype": "int64",
                             "vocabulary": info["vocabulary"]}]

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
        {"name": "x", "shape": [4, 7], "dtype": "float32",
         "dataset_shape": [None, 7], "vocabulary": None}
    ]
    assert body["target_kind"] is None
    assert body["outputs"] == 3 and body["task"] == "classification"
    assert body["classes"] == ["a", "b", "c"]
    assert body["path"] == str(dataset.resolve())

    # 知らない版は ML-EDA の誤りの文で 422, 説明が無い場合は 404
    other = write_dataset(tmp_path / "ds9", 9)
    response = client.post("/api/dataset/inputs", json={"path": str(other)})
    assert response.status_code == 422
    assert "format_version 9" in response.json()["detail"]
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


def test_dataset_inputs_items(client: TestClient, tmp_path: Path) -> None:
    """
    本物の ML-EDA で書き出した画像と領域のデータセット (版 3) から,
    入力の形とクラスの数を読むことを確かめる関数

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
    # ファイルの組の model_inputs のある ML-EDA だけ
    ml_eda = pytest.importorskip("ml_eda")
    if not hasattr(ml_eda.pipeline, "items_model_inputs"):
        pytest.skip("ML-EDA の model_inputs がファイルの組を読みません")

    # 32 x 32 にそろえた画像と, 値 0, 1, 2 の領域の説明 (版 3. model_inputs
    # は説明だけを読む)
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "dataset.yaml").write_text(yaml.safe_dump({
        "format_version": 3, "role": "predict",
        "target": {"kind": "mask", "entries": ["mask"]},
        "entries": {
            "image": {"kind": "image", "role": "input",
                      "shape": [32, 32, 3], "dtype": "uint8",
                      "steps": [{"op": "resize", "width": 32,
                                 "height": 32}]},
            "mask": {"kind": "mask", "role": "target", "shape": [32, 32],
                     "dtype": "uint8", "values": [0, 1, 2],
                     "classes": [{"value": 1, "name": "circle"},
                                 {"value": 2, "name": "square"}]},
        },
    }))

    # 入力は (バッチ, チャンネル, 高さ, 幅), 出す値は背景を含むクラスの数
    response = client.post(
        "/api/dataset/inputs", json={"path": str(tmp_path / "ds")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputs"] == [{
        "name": "image", "shape": [4, 3, 32, 32], "dtype": "float32",
        "dataset_shape": [None, 3, 32, 32], "vocabulary": None
    }]
    assert body["target_kind"] == "mask" and body["task"] is None
    assert body["outputs"] == 3 and len(body["classes"]) == 3


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
    config.write_text(
        "data:\n  dataset: outputs/dataset\n"
        "model:\n  spec: models/model.yaml\n"
    )

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

    # データセットの候補は config の data.dataset (プロジェクトから)
    assert target["dataset"] == str(project.resolve() / "outputs" / "dataset")

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


def test_project_named_target(client: TestClient, tmp_path: Path) -> None:
    """
    models/ の下の名前で保存先を選び, config を書き換えず, 規則に合わない
    名前とリンクを断ることを確かめる関数

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
    # 名前を書くと models/<名前>.yaml. 空白だけなら既定の保存先
    project = make_project_dir(tmp_path / "proj")
    config = project / "configs" / "default" / "train.yaml"
    before = config.read_text()
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "mlp"}
    )
    assert response.status_code == 200, response.text
    target = response.json()
    expected = project.resolve() / "models" / "mlp.yaml"
    assert target["path"] == str(expected)
    assert target["source"] == "name" and target["exists"] is False
    assert target["dataset"] == str(project.resolve() / "outputs" / "dataset")
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "  "}
    )
    assert response.json()["source"] == "project"

    # 2 つの候補を並べて保存でき, config の model.spec は変わらない
    for name in ("mlp", "wide"):
        response = client.put(
            "/api/project/model",
            json={"path": str(project), "name": name, "spec": SPEC}
        )
        assert response.status_code == 200, response.text
    assert sorted(item.name for item in (project / "models").iterdir()) == [
        "mlp.yaml", "wide.yaml"
    ]
    assert config.read_text() == before

    # 既にある名前は上書きの許しが要る (409)
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "name": "mlp", "spec": SPEC}
    )
    assert response.status_code == 409
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "mlp"}
    )
    assert response.json()["exists"] is True

    # 規則に合わない名前 (中のディレクトリ, 外, 拡張子, 隠しファイル) は 422
    for name in ("a/b", "../x", "x.yaml", ".hidden", "a" * 65):
        response = client.put(
            "/api/project/model",
            json={"path": str(project), "name": name, "spec": SPEC}
        )
        assert response.status_code == 422, name
    assert not (tmp_path / "x.yaml").exists()

    # models/ のファイルがリンクなら 422 (中を指すリンクも)
    (project / "models" / "link.yaml").symlink_to(
        project / "models" / "wide.yaml"
    )
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "name": "link", "spec": SPEC,
              "overwrite": True}
    )
    assert response.status_code == 422
    assert "リンク" in response.json()["detail"]

    # models/ がリンクなら 422 (外のディレクトリに書かない)
    other = make_project_dir(tmp_path / "proj2")
    outside = tmp_path / "outside"
    outside.mkdir()
    (other / "models").symlink_to(outside, target_is_directory=True)
    response = client.put(
        "/api/project/model",
        json={"path": str(other), "name": "mlp", "spec": SPEC}
    )
    assert response.status_code == 422
    assert not list(outside.iterdir())


def test_project_named_sibling(client: TestClient, tmp_path: Path) -> None:
    """
    同じ名前の .yml か, 大文字と小文字だけ違うファイルがあればそこに書き,
    既定の保存先と同じ名前に印を付けることを確かめる関数

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
    # a.yml があれば名前 a は a.yml (既にある扱い. a.yaml は作らない)
    project = make_project_dir(tmp_path / "proj").resolve()
    models = project / "models"
    models.mkdir()
    (models / "a.yml").write_text("old\n")
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "a"}
    )
    assert response.status_code == 200, response.text
    target = response.json()
    assert target["path"] == str(models / "a.yml") and target["exists"]
    assert target["default"] is False
    body = {"path": str(project), "name": "a", "spec": SPEC}
    assert client.put("/api/project/model", json=body).status_code == 409
    response = client.put(
        "/api/project/model", json={**body, "overwrite": True}
    )
    assert response.status_code == 200, response.text
    assert sorted(item.name for item in models.iterdir()) == ["a.yml"]
    assert load_spec(models / "a.yml")["nodes"]["fc"]["unit"] == "nn.Linear"

    # 大文字と小文字だけ違うファイルは, 区別しないファイルシステムなら実際の
    # 名前で出す (区別するなら別のファイル)
    (models / "Wide.yaml").write_text("old\n")
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "wide"}
    )
    target = response.json()
    if (models / "wide.yaml").exists():
        assert target["path"] == str(models / "Wide.yaml")
        assert target["exists"] is True
    else:
        assert target["path"] == str(models / "wide.yaml")
        assert target["exists"] is False

    # config の model.spec と同じ名前は既定の保存先と同じ印
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "model"}
    )
    assert response.json()["default"] is True


def test_project_save_bound_target(
    client: TestClient, tmp_path: Path
) -> None:
    """
    確かめた保存先と, 書くときに決め直した保存先が違えば, 上書きの許しが
    あっても書かない (409) ことを確かめる関数

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
    # a.yml を確かめた後に, 他から a.yaml が作られた場合
    project = make_project_dir(tmp_path / "proj").resolve()
    models = project / "models"
    models.mkdir()
    (models / "a.yml").write_text("old\n")
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "a"}
    )
    checked = response.json()["path"]
    assert checked == str(models / "a.yml")
    (models / "a.yaml").write_text("other\n")
    body = {
        "path": str(project), "name": "a", "spec": SPEC, "overwrite": True,
        "target": checked
    }
    response = client.put("/api/project/model", json=body)
    assert response.status_code == 409
    assert "変わりました" in response.json()["detail"]
    assert (models / "a.yaml").read_text() == "other\n"
    assert (models / "a.yml").read_text() == "old\n"

    # 確かめ直した保存先なら書く
    response = client.put(
        "/api/project/model",
        json={**body, "target": str(models / "a.yaml")}
    )
    assert response.status_code == 200, response.text
    assert load_spec(models / "a.yaml")["nodes"]["fc"]["unit"] == "nn.Linear"

    # 確かめた後に config の model.spec が変わった場合も同じ
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    checked = response.json()["path"]
    assert checked == str(models / "model.yaml")
    (project / "configs" / "default" / "train.yaml").write_text(
        "model:\n  spec: models/a.yaml\n"
    )
    body = {"path": str(project), "spec": SPEC, "target": checked}
    response = client.put("/api/project/model", json=body)
    assert response.status_code == 409
    assert not (models / "model.yaml").exists()


def test_same_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    まだ無いファイルの同じかの判定が, 大文字と小文字を区別しない置き場で
    大小だけ違う名前を同じとみなすことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        置き場の見分けを替える

    Returns
    ----------
    None
    """
    # 置き場の見分けは, 実際に大小を入れ替えた名前で探した結果と同じ
    folder = tmp_path / "Models"
    folder.mkdir()
    swapped = tmp_path / "mODELS"
    assert ignores_case(folder / "a.yaml") == swapped.exists()

    # 両方無く大小だけ違う名前は, 置き場の見分けで決める
    first, second = folder / "Model.yaml", folder / "model.yaml"
    monkeypatch.setattr(project_module, "ignores_case", lambda path: True)
    assert same_file(first, second) is True
    monkeypatch.setattr(project_module, "ignores_case", lambda path: False)
    assert same_file(first, second) is False

    # 名前の違うものと, 片方だけあるものは区別する置き場の別のファイル
    monkeypatch.setattr(project_module, "ignores_case", lambda path: True)
    assert same_file(folder / "a.yaml", second) is False
    first.write_text("x\n")
    if not second.exists():
        assert same_file(first, second) is False


def test_project_upper_suffix(client: TestClient, tmp_path: Path) -> None:
    """
    大文字の拡張子の保存先 (ML-Dashboard の候補に出ない) を断ることを
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
    # config の model.spec の拡張子が大文字
    project = make_project_dir(tmp_path / "proj").resolve()
    (project / "configs" / "default" / "train.yaml").write_text(
        "model:\n  spec: models/Net.YAML\n"
    )
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 422
    assert "小文字" in response.json()["detail"]
    response = client.put(
        "/api/project/model", json={"path": str(project), "spec": SPEC}
    )
    assert response.status_code == 422
    assert not (project / "models").exists()

    # 名前の保存先が大文字の拡張子の既にあるファイルになる場合 (区別しない
    # 置き場) も断り, 区別する置き場なら小文字の別のファイル
    models = project / "models"
    models.mkdir()
    (models / "wide.YML").write_text("old\n")
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "wide"}
    )
    if (models / "wide.yml").exists():
        assert response.status_code == 422
        assert "小文字" in response.json()["detail"]
    else:
        assert response.status_code == 200, response.text
        assert response.json()["path"] == str(models / "wide.yaml")


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
    assert target["problem_path"] == str(problem.resolve())
    assert target["example"] is None
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
    assert response.json()["problem_path"] == str(project / "problem.yaml")

    # 名前を書くと, 記録のプロジェクトの models/<名前>.yaml
    response = client.post(
        "/api/project/target", json={"path": str(problem), "name": "cnn"}
    )
    assert response.json()["path"] == str(project / "models" / "cnn.yaml")
    assert response.json()["problem"] == "sales"

    # 受け持ちに合う雛形の名前を渡す (名前の保存先でも). 同梱の例に無い名前と
    # 名前でない値は渡さない (開けない雛形を出さないため)
    record["model"]["example"] = "unet_segmentation"
    for name in (None, "cnn"):
        response = client.post(
            "/api/project/target", json={"path": str(problem), "name": name}
        )
        assert response.json()["example"] == "unet_segmentation"
    for value in ("no_such_example", 3, ["unet_segmentation"]):
        record["model"]["example"] = value
        response = client.post(
            "/api/project/target", json={"path": str(problem)}
        )
        assert response.status_code == 200, response.text
        assert response.json()["example"] is None
    record["model"].pop("example")

    # プロジェクトの外を指す記録は 422
    record["model"]["project_model_spec"] = str(tmp_path / "other.yaml")
    response = client.post(
        "/api/project/target", json={"path": str(problem)}
    )
    assert response.status_code == 422
    assert "外" in response.json()["detail"]
    response = client.post(
        "/api/project/target", json={"path": str(problem), "name": "cnn"}
    )
    assert response.status_code == 422

    # 記録の無い問題の設定のファイルは 422. プロジェクトの中の記録の無い
    # problem.yaml は骨組みの既定
    record["model"] = {}
    response = client.post(
        "/api/project/target", json={"path": str(problem)}
    )
    assert response.status_code == 422
    assert "書き出してありますが" in response.json()["detail"]
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


def test_dataset_inputs_odd_values(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    数の目的の列の名前は文字列にし, 形の崩れた値は 422 にすることを
    確かめる関数

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
    # 目的の列の名前が数, クラスがリストでない結果
    result: dict[str, Any] = {
        "inputs": [{"name": "x", "shape": [None, 2], "dtype": "float32"}],
        "task": "regression", "target": 0, "classes": "a",
        "outputs": 1
    }
    monkeypatch.setitem(
        sys.modules, "ml_eda", SimpleNamespace(model_inputs=StubInputs(result))
    )
    dataset = write_dataset(tmp_path / "ds", 1)
    response = client.post(
        "/api/dataset/inputs", json={"path": str(dataset)}
    )
    assert response.status_code == 200, response.text
    assert response.json()["target"] == "0"
    assert response.json()["classes"] is None

    # 形が数でない場合は 422
    result["inputs"] = [{"name": "x", "shape": ["a"], "dtype": "float32"}]
    response = client.post(
        "/api/dataset/inputs", json={"path": str(dataset)}
    )
    assert response.status_code == 422


class StubInputs:
    def __init__(self, result: dict[str, Any]):
        # 返す結果を保存
        self.result = result

    def __call__(self, path: Any) -> dict[str, Any]:
        """
        決まった結果を返す関数

        Parameters
        ----------
        path: Any
            データセットのディレクトリ (使わない)

        Returns
        ----------
        result: dict[str, Any]
            保存した結果
        """
        result = self.result

        return result


def test_project_symlink_and_mode(client: TestClient, tmp_path: Path) -> None:
    """
    プロジェクトの外を指す models のリンクを断り, 書いたファイルが普通の
    権限になることを確かめる関数

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
    # 外を指すリンクの models は 422
    project = make_project_dir(tmp_path / "proj")
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "models").symlink_to(outside, target_is_directory=True)
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 422
    assert "外" in response.json()["detail"]

    # 普通のプロジェクトでは, 本人以外も読める権限で書く
    other = make_project_dir(tmp_path / "proj2")
    response = client.put(
        "/api/project/model", json={"path": str(other), "spec": SPEC}
    )
    assert response.status_code == 200
    mode = (other / "models" / "model.yaml").stat().st_mode & 0o777
    assert mode & 0o044


@pytest.mark.parametrize(
    "text",
    ["model:\n  spec: models/model.yaml\n", "data: [1, 2\n", "data: 3\n"]
)
def test_project_dir_without_dataset(
    client: TestClient, tmp_path: Path, text: str
) -> None:
    """
    config に data.dataset が無いか読めない記録の無いプロジェクトでは,
    データセットの候補を出さずに保存先を返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    text: str
        config の中身

    Returns
    ----------
    None
    """
    project = make_project_dir(tmp_path / "proj")
    (project / "configs" / "default" / "train.yaml").write_text(text)
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 200, response.text
    assert response.json()["dataset"] is None


def test_problem_record_guards(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    記録のプロジェクトが学習のプロジェクトでない場合と, 消したプロジェ
    クトの場合に書かないこと, 記録の無いプロジェクトでは config の
    model.spec に書くことを確かめる関数

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
    # config の無い場所を指す記録 (手で書いた問題の設定など) は 422
    other = (tmp_path / "home").resolve()
    (other / ".config").mkdir(parents=True)
    victim = other / ".config" / "x.yaml"
    victim.write_text("keep: true\n")
    problem = tmp_path / "store" / "p.yaml"
    problem.parent.mkdir()
    problem.write_text("name: p\n")
    record = {
        "model": {"project_model_spec": str(victim)},
        "data": {},
        "project": {"path": str(other)}
    }
    monkeypatch.setitem(sys.modules, "ml_quest", StubQuest(record))
    response = client.put(
        "/api/project/model",
        json={"path": str(problem), "spec": SPEC, "overwrite": True}
    )
    assert response.status_code == 422
    assert victim.read_text() == "keep: true\n"

    # 消したプロジェクトの場所を作り直さない
    gone = (tmp_path / "gone").resolve()
    record["model"] = {"project_model_spec": str(gone / "models" / "m.yaml")}
    record["project"] = {"path": str(gone)}
    response = client.put(
        "/api/project/model", json={"path": str(problem), "spec": SPEC}
    )
    assert response.status_code == 422 and not gone.exists()

    # 記録の無いプロジェクトは config の model.spec に書く
    project = make_project_dir(tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    config.write_text(
        "data:\n  dataset: outputs/dataset\n"
        "model:\n  spec: models/other.yaml\n"
    )
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.json()["path"] == str(project / "models" / "other.yaml")


@pytest.mark.parametrize(
    "spec", [
        "configs/default/train.yaml", "configs/other.yaml",
        "Configs/new/model.yaml"
    ]
)
def test_project_refuses_configs(
    client: TestClient, tmp_path: Path, spec: str
) -> None:
    """
    config の model.spec が学習の設定のディレクトリ (configs/) を指す
    場合に, 確かめも保存も断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    spec: str
        config の model.spec

    Returns
    ----------
    None
    """
    # config そのものも, configs/ の下の別のファイルも 422 で書かない
    project = make_project_dir(tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    text = f"data:\n  dataset: outputs/dataset\nmodel:\n  spec: {spec}\n"
    config.write_text(text)
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 422
    assert "configs/" in response.json()["detail"]
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "spec": SPEC, "overwrite": True}
    )
    assert response.status_code == 422
    assert config.read_text() == text
    assert sorted(path.name for path in project.iterdir()) == ["configs"]


@pytest.mark.parametrize(
    "spec, word", [
        ("problem.yaml", "problem.yaml"), ("Problem.YAML", "problem.yaml"),
        ("data/model.yaml", "data/"), ("DATA/new/model.yaml", "data/"),
        ("outputs/dataset/dataset.yaml", "outputs/"),
        ("outputs/model.yml", "outputs/")
    ]
)
def test_project_refuses_reserved(
    client: TestClient, tmp_path: Path, spec: str, word: str
) -> None:
    """
    config の model.spec がプロジェクトの直下の problem.yaml か, data/
    か outputs/ の中を指す場合に, 確かめも保存も断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    spec: str
        config の model.spec
    word: str
        断る理由に出る名前

    Returns
    ----------
    None
    """
    # 確かめの時点で理由を付けて断り, 何も書かない
    project = make_project_dir(tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    config.write_text(f"model:\n  spec: {spec}\n")
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 422
    assert word in response.json()["detail"]
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "spec": SPEC, "overwrite": True}
    )
    assert response.status_code == 422
    assert sorted(path.name for path in project.iterdir()) == ["configs"]


def test_project_reserved_names_elsewhere(
    client: TestClient, tmp_path: Path
) -> None:
    """
    直下でない problem.yaml と, data か outputs で始まる別の名前には
    書けることを確かめる関数

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
    # 断るのは直下の problem.yaml と, data/ と outputs/ の中だけ
    project = make_project_dir(tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    for spec in ("models/problem.yaml", "data.yaml", "outputs2/m.yaml"):
        config.write_text(f"model:\n  spec: {spec}\n")
        response = client.put(
            "/api/project/model", json={"path": str(project), "spec": SPEC}
        )
        assert response.status_code == 200, response.text
        assert (project / spec).is_file()


@pytest.mark.parametrize(
    "spec, blocker", [
        ("models/a/model.yaml", "models"),
        ("models/a/b/model.yaml", "models/a"),
    ]
)
def test_project_parent_is_file(
    client: TestClient, tmp_path: Path, spec: str, blocker: str
) -> None:
    """
    保存先までの途中の名前がファイルの場合に, 確かめの時点で断ることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    spec: str
        config の model.spec
    blocker: str
        途中に置くファイル

    Returns
    ----------
    None
    """
    # 途中のファイルを理由に出し, 保存も 500 でなく 422 にする
    project = make_project_dir(tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    config.write_text(f"model:\n  spec: {spec}\n")
    (project / blocker).parent.mkdir(parents=True, exist_ok=True)
    (project / blocker).write_text("x")
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 422
    assert str(project / blocker) in response.json()["detail"]
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "spec": SPEC, "overwrite": True}
    )
    assert response.status_code == 422
    assert (project / blocker).read_text() == "x"

    # 壊れたリンクも同じ
    (project / blocker).unlink()
    (project / blocker).symlink_to(tmp_path / "missing")
    response = client.post(
        "/api/project/target", json={"path": str(project)}
    )
    assert response.status_code == 422


def test_project_named_models_file(
    client: TestClient, tmp_path: Path
) -> None:
    """
    名前で選ぶ保存先で models がファイルの場合に, 確かめの時点で断る
    ことを確かめる関数

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
    # models/<名前>.yaml の途中の models がファイル
    project = make_project_dir(tmp_path / "proj").resolve()
    (project / "models").write_text("x")
    response = client.post(
        "/api/project/target", json={"path": str(project), "name": "cand"}
    )
    assert response.status_code == 422
    assert "ディレクトリではありません" in response.json()["detail"]


def test_project_deep_spec(client: TestClient, tmp_path: Path) -> None:
    """
    2 段以上の深さの保存先では, プロジェクトの中の無いディレクトリだけを
    作り, 外には作らないことを確かめる関数

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
    # models/a/b/ を順に作って書く
    project = make_project_dir(tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    config.write_text("model:\n  spec: models/a/b/model.yaml\n")
    response = client.put(
        "/api/project/model", json={"path": str(project), "spec": SPEC}
    )
    assert response.status_code == 200, response.text
    expected = project / "models" / "a" / "b" / "model.yaml"
    assert response.json()["path"] == str(expected)
    assert load_spec(expected)["nodes"]["fc"]["unit"] == "nn.Linear"

    # プロジェクトの外の場所と, 無いプロジェクトには作らない
    outside = tmp_path / "outside" / "x" / "m.yaml"
    with pytest.raises(HTTPException) as caught:
        make_parents(outside, str(project))
    assert caught.value.status_code == 422
    gone = tmp_path / "gone"
    with pytest.raises(HTTPException):
        make_parents(gone / "models" / "m.yaml", str(gone))
    assert not (tmp_path / "outside").exists() and not gone.exists()


def save_problem(
    quest: Any, home: Path, name: str, problem: dict[str, Any]
) -> Path:
    """
    問題の設定を, ML-Pathfinder の置き場の形で保存する関数

    Parameters
    ----------
    quest: Any
        ML-Quest のモジュール
    home: Path
        ML-Pathfinder の置き場
    name: str
        問題の設定の識別子 (ディレクトリの名前)
    problem: dict[str, Any]
        問題の設定

    Returns
    ----------
    path: Path
        保存した problem.yaml
    """
    path = home / "problems" / name / "problem.yaml"
    quest.save_problem(problem, path)

    return path


def test_list_problems(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ML-Pathfinder の置き場の問題の設定を並べ, プロジェクトを書き出した
    ものだけを使えるとし, 他は理由を付けることを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        環境変数の差し替え

    Returns
    ----------
    None
    """
    # 置き場が無ければ空
    quest = pytest.importorskip("ml_quest")
    home = tmp_path / "pathfinder"
    monkeypatch.setenv("ML_PATHFINDER_HOME", str(home))
    response = client.get("/api/problems")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "root": str(home.resolve() / "problems"), "problems": [],
        "error": None
    }

    # 書き出したもの, 書き出していないもの, 読めないもの, 問題の設定で
    # ないもの
    problem = quest.example_problem(next(iter(quest.EXAMPLES)))
    project = tmp_path / "proj"
    quest.make_project(problem, project)
    recorded = save_problem(
        quest, home, "a", quest.record_project(problem, project)
    )
    save_problem(quest, home, "b", problem)
    no_spec = quest.record_project(problem, project)
    no_spec["project"]["model_spec"] = None
    save_problem(quest, home, "f", no_spec)
    broken = home / "problems" / "c" / "problem.yaml"
    broken.parent.mkdir()
    broken.write_text("name: [\n")
    other = home / "problems" / "d" / "problem.yaml"
    other.parent.mkdir()
    other.write_text("format_version: 2\n")
    (home / "problems" / "e").mkdir()
    response = client.get("/api/problems")
    assert response.status_code == 200, response.text
    entries = {
        Path(entry["path"]).parent.name: entry
        for entry in response.json()["problems"]
    }
    assert sorted(entries) == ["a", "b", "c", "d", "f"]

    # 書き出したものは保存先, プロジェクト, データセットの場所を持ち,
    # 「プロジェクトに保存」で確かめるのと同じ保存先
    first = entries["a"]
    assert first["usable"] is True and first["reason"] is None
    assert first["name"] == problem["name"] and first["created_at"]
    assert first["project"] == str(project.resolve())
    assert first["dataset"] == str(project.resolve() / "outputs" / "dataset")
    response = client.post(
        "/api/project/target", json={"path": str(recorded)}
    )
    assert response.json()["path"] == first["target"]
    assert first["target"].endswith("models/model.yaml")

    # 他は使えず, 理由が付く
    assert entries["b"]["usable"] is False
    assert "書き出していません" in entries["b"]["reason"]

    # 書き出してあり, 定義の置き場の記録だけが無いものは, その理由
    assert entries["f"]["usable"] is False
    assert "書き出していません" not in entries["f"]["reason"]
    assert "書き出してありますが" in entries["f"]["reason"]
    assert str(project.resolve()) in entries["f"]["reason"]
    response = client.post(
        "/api/project/target",
        json={"path": entries["f"]["path"]}
    )
    assert response.status_code == 422
    assert "書き出してありますが" in response.json()["detail"]
    response = client.post(
        "/api/project/target", json={"path": entries["b"]["path"]}
    )
    assert response.status_code == 422
    assert "書き出していません" in response.json()["detail"]
    assert "読めません" in entries["c"]["reason"]
    assert "問題の設定のファイルではありません" in entries["d"]["reason"]
    assert entries["d"]["name"] == "d"

    # 記録のプロジェクトを消すと, 使えなくなり理由が付く
    (project / "configs" / "default" / "train.yaml").unlink()
    response = client.get("/api/problems")
    first = next(
        entry for entry in response.json()["problems"]
        if entry["path"] == str(recorded)
    )
    assert first["usable"] is False and "ML-Pathfinder" in first["reason"]


def rank_problem(quest: Any) -> dict[str, Any]:
    """
    見本の課題から, 並べる受け持ちを一から学習する問題の設定を作る関数

    Parameters
    ----------
    quest: Any
        ML-Quest のモジュール

    Returns
    ----------
    problem: dict[str, Any]
        問題の設定 (雛形は two_tower_retrieval. 反応の表の列は member,
        item_id, ts, qty)
    """
    # 見本の受け持ちを作り直す (ML-Quest のテストの受け持ちの入れ替えと同じ)
    problem = quest.example_problem(next(iter(quest.EXAMPLES)))
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = quest.answer(
        problem, "information.columns", ["member", "item_id", "ts", "qty"]
    )
    answers = [
        ("role", "rank"), ("what", "試しの出力"), ("item", "1 件"),
        ("baseline", "今のやり方"), ("evaluation.link", "良くなると損が減る"),
        ("unit", "query"), ("output", "ranking"),
        ("target.source", "feedback"), ("method", "scratch"),
        ("evaluation.metric", "ndcg"), ("rank.user", "member"),
        ("rank.item", "item_id"), ("rank.time", "ts"),
        ("rank.reaction", "qty"), ("rank.k", 5)
    ]
    for key, value in answers:
        problem = quest.answer(problem, f"ml.parts.0.{key}", value)

    return problem


def test_problem_example(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    本物の ML-Quest の受け渡しの雛形の名前を, 一覧と保存先に渡すことを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        環境変数の差し替え

    Returns
    ----------
    None
    """
    # 並べるを一から学習するものと, 見本 (表の列の予測. 雛形無し)
    quest = pytest.importorskip("ml_quest")
    home = tmp_path / "pathfinder"
    monkeypatch.setenv("ML_PATHFINDER_HOME", str(home))
    example = quest.example_problem(next(iter(quest.EXAMPLES)))
    project = tmp_path / "proj"
    quest.make_project(example, project)
    rank = save_problem(
        quest, home, "rank",
        quest.record_project(rank_problem(quest), project)
    )
    save_problem(quest, home, "column", quest.record_project(example, project))
    response = client.get("/api/problems")
    assert response.status_code == 200, response.text
    entries = {
        Path(entry["path"]).parent.name: entry
        for entry in response.json()["problems"]
    }
    assert entries["rank"]["usable"] is True
    assert entries["rank"]["example"] == "two_tower_retrieval"
    assert entries["column"]["example"] is None

    # 保存先の確かめでも同じ名前 (保存先と出力の大きさの受け取りは変えない)
    response = client.post("/api/project/target", json={"path": str(rank)})
    assert response.status_code == 200, response.text
    target = response.json()
    assert target["example"] == "two_tower_retrieval"
    assert target["path"] == entries["rank"]["target"]
    assert target["source"] == "problem" and target["exists"] is False


def test_positive_int() -> None:
    """
    語彙の大きさなどの値を, 正の整数だけ通すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正の整数だけ. 真偽, 0, 負, 小数, 文字, None は None
    assert positive_int(21) == 21
    for value in (True, 0, -3, 2.0, "21", None):
        assert positive_int(value) is None


def test_dataset_inputs_vocabulary(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    文の語の番号の入力の語彙の大きさを応答に通し, 正の整数でなければ null
    にすることを, 代わりの ML-EDA で確かめる関数

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
    # 代わりの ML-EDA と, 語彙の大きさの値ごとの応答
    monkeypatch.setitem(
        sys.modules, "ml_eda", SimpleNamespace(model_inputs=stub_inputs)
    )
    for index, (value, expected) in enumerate(
        [(21, 21), ("21", None), (0, None), (True, None)]
    ):
        dataset = write_dataset(tmp_path / f"text{index}", 1)
        (dataset / "dataset.yaml").write_text(
            yaml.safe_dump({"format_version": 1, "vocabulary": value})
        )
        response = client.post(
            "/api/dataset/inputs", json={"path": str(dataset)}
        )
        assert response.status_code == 200, response.text
        item = response.json()["inputs"][0]
        assert item["dtype"] == "int64" and item["shape"] == [4, 16]
        assert item["vocabulary"] == expected


def test_dataset_inputs_text_vocabulary(
    client: TestClient, tmp_path: Path
) -> None:
    """
    本物の ML-EDA で書き出した文のデータセットから, 語彙の大きさが API の
    応答に出ることを確かめる関数

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
    # 固定した ML-EDA (語彙の大きさを返す版)
    ml_eda = pytest.importorskip("ml_eda")

    # 2 クラスの短い文を語の番号の並びで書き出す
    for label, words in (("pos", "good great"), ("neg", "bad awful")):
        (tmp_path / "texts" / label).mkdir(parents=True)
        for index in range(5):
            (tmp_path / "texts" / label / f"{index}.txt").write_text(
                f"this is {words} {index}", encoding="utf-8"
            )
    recipe = {
        "target": "label", "task": "classification",
        "item_steps": [{"op": "vocabulary", "length": 6}],
        "split": {"method": "stratified", "valid": 0.2},
    }
    dataset = tmp_path / "ds"
    ml_eda.export_dataset(ml_eda.load_data(tmp_path / "texts"), recipe,
                          dataset)
    steps = ml_eda.load_dataset(dataset).recipe["item_steps"]
    size = len(steps[0]["state"]["vocabulary"])

    # 応答の入力は語の番号 (int64) で, 語彙の大きさ付き
    response = client.post("/api/dataset/inputs", json={"path": str(dataset)})
    assert response.status_code == 200, response.text
    item = response.json()["inputs"][0]
    assert item["dtype"] == "int64" and item["vocabulary"] == size
    # 埋め草と語彙に無い語と, 学習用の文の語 (共通の this, is, クラスの語,
    # 番号) の数の範囲
    assert 2 + 2 < size <= 2 + 2 + 4 + 5


def table_generation_problem(ml_quest: Any) -> dict[str, Any]:
    """
    見本の課題から, 表の生成 (作るで正解の種類が表) の問題の設定を作る関数

    Parameters
    ----------
    ml_quest: Any
        ML-Quest のモジュール

    Returns
    ----------
    problem: dict[str, Any]
        受け持ちを表の生成に答え直した問題の設定
    """
    # 受け持ちの答えだけを作り直す
    problem = ml_quest.example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key, value in [("role", "generate"), ("unit", "row"),
                       ("output", "table"), ("target.source", "self"),
                       ("target.kind", "table"), ("method", "scratch")]:
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)

    return problem


def test_dataset_inputs_table_generation(
    client: TestClient, tmp_path: Path
) -> None:
    """
    表の生成の問題の設定を渡すと, 入力と出す値の数を符号化した行の幅に
    することを確かめる関数 (ML-EDA と table_encoding のある ML-Quest が
    無ければ飛ばす)

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
    # 読み込みに要るライブラリ
    ml_eda = pytest.importorskip("ml_eda")
    ml_quest = pytest.importorskip("ml_quest")
    if not hasattr(ml_quest, "table_encoding"):
        pytest.skip("ML-Quest に table_encoding がありません")

    # 数の列 2 つと, 値の種類 3 と 2 の種類の列 (列の数 4, 幅 7) を, 目的の
    # 列の無いレシピで書き出し
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "age": rng.normal(40, 10, 60), "visits": rng.integers(0, 9, 60),
        "city": np.array(["a", "b", "c"] * 20),
        "member": np.array(["yes", "no"] * 30)
    })
    source = tmp_path / "table.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "task": "none", "steps": [],
        "split": {"method": "random", "valid": 0.2, "test": 0.2}
    }
    dataset = tmp_path / "ds"
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, dataset)
    problem = tmp_path / "problem.yaml"
    ml_quest.save_problem(table_generation_problem(ml_quest), problem)

    # 問題の設定を渡すと, 入力の列の次元と出す値の数が符号化した行の幅
    response = client.post(
        "/api/dataset/inputs",
        json={"path": str(dataset), "problem": str(problem)}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["inputs"][0]["shape"] == [4, 7]
    assert body["inputs"][0]["dataset_shape"] == [None, 7]
    assert body["outputs"] == 7
    assert body["table"] == {
        "width": 7, "numbers": 2, "categories": 2, "values": 5,
        "columns": 4, "problem": "総菜の作る量"
    }

    # 渡さないか, 表の生成でない問題の設定なら列の数のまま
    other = tmp_path / "other.yaml"
    ml_quest.save_problem(ml_quest.example_problem("総菜の作る量"), other)
    for extra in ({}, {"problem": str(other)}):
        response = client.post(
            "/api/dataset/inputs", json={"path": str(dataset), **extra}
        )
        assert response.status_code == 200, response.text
        assert response.json()["inputs"][0]["shape"] == [4, 4]
        assert response.json()["table"] is None

    # 問題の設定が無い場所は 404, 幅を求められない (目的の列のある)
    # データセットは 422
    response = client.post(
        "/api/dataset/inputs",
        json={"path": str(dataset), "problem": str(tmp_path / "none.yaml")}
    )
    assert response.status_code == 404
    labeled = tmp_path / "labeled"
    ml_eda.export_dataset(
        ml_eda.load_data(source),
        {**recipe, "task": "regression", "target": "age"}, labeled
    )
    response = client.post(
        "/api/dataset/inputs",
        json={"path": str(labeled), "problem": str(problem)}
    )
    assert response.status_code == 422
    assert "目的の列" in response.json()["detail"]


def test_dataset_inputs_rank(client: TestClient, tmp_path: Path) -> None:
    """
    並べるの問題の設定と反応の表を渡すと, 入力 user と item の大きさを
    学習用の利用者と品物の数にし, 割合はプロジェクトの config から読む
    ことを確かめる関数 (ML-EDA と rank_encoding のある ML-Quest が無ければ
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
    # 読み込みに要るライブラリ
    pytest.importorskip("ml_eda")
    ml_quest = pytest.importorskip("ml_quest")
    if not hasattr(ml_quest, "rank_encoding"):
        pytest.skip("ML-Quest に rank_encoding がありません")

    # 利用者 6 人と品物 4 つの 40 行 (時刻の順で, 新しい 4 行だけに出る
    # 利用者 m9 と品物 i9. 反応の強さ 0 の行を 1 つ)
    rows = [
        {"member": f"m{number % 6}", "item_id": f"i{number % 4}",
         "ts": number, "qty": 0 if number == 3 else 1}
        for number in range(36)
    ] + [{"member": "m9", "item_id": "i9", "ts": 36 + number, "qty": 1}
         for number in range(4)]
    source = tmp_path / "reactions.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    problem = ml_quest.attach_data(
        rank_problem(ml_quest), str(source), copy_to=str(tmp_path / "copies")
    )
    project = tmp_path / "proj"
    ml_quest.make_project(problem, project)
    problem = ml_quest.record_project(problem, project)
    place = tmp_path / "problem.yaml"
    ml_quest.save_problem(problem, place)
    dataset = problem["project"]["dataset"]

    # 骨組みの config の割合 (検証用とテスト用が 1 割ずつ) では m9 と i9
    # は学習用にいない
    response = client.post(
        "/api/dataset/inputs", json={"path": dataset, "problem": str(place)}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [(item["name"], item["shape"], item["dataset_shape"])
            for item in body["inputs"]] == [
        ("user", [4, 6], [None, 6]), ("item", [4, 4], [None, 4])
    ]
    assert body["outputs"] is None and body["table"] is None
    rank = body["rank"]
    assert (rank["users"], rank["items"]) == (6, 4)
    assert (rank["valid"], rank["test"]) == (0.1, 0.1)
    assert sum(rank["rows"].values()) == 39
    assert rank["problem"] == problem["name"]
    assert any("反応の強さ" in note for note in rank["notes"])

    # config の割合を小さくすると, 学習用に m9 と i9 も入る
    config_path = project / "configs" / "default" / "train.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"].update(valid=0.02, test=0.0)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    response = client.post(
        "/api/dataset/inputs", json={"path": dataset, "problem": str(place)}
    )
    assert response.status_code == 200, response.text
    rank = response.json()["rank"]
    assert (rank["users"], rank["items"]) == (7, 5)
    assert (rank["valid"], rank["test"]) == (0.02, 0.0)
    assert set(rank["rows"]) == {"train", "valid"}

    # config に割合が無ければ学習のスクリプトの既定 (無し と返す)
    del config["data"]["valid"], config["data"]["test"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    response = client.post(
        "/api/dataset/inputs", json={"path": dataset, "problem": str(place)}
    )
    assert response.status_code == 200, response.text
    rank = response.json()["rank"]
    assert (rank["users"], rank["items"]) == (6, 4)
    assert (rank["valid"], rank["test"]) == (None, None)

    # 反応の表の無い場所は 404, 分けられない割合は 422
    response = client.post(
        "/api/dataset/inputs",
        json={"path": str(tmp_path / "none.csv"), "problem": str(place)}
    )
    assert response.status_code == 404
    config["data"].update(valid=0.0)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    response = client.post(
        "/api/dataset/inputs", json={"path": dataset, "problem": str(place)}
    )
    assert response.status_code == 422
    assert "data.valid" in response.json()["detail"]

    # 学習の設定が壊れていて読めなければ, 既定の割合に戻して知らせる
    config_path.write_text("data: [", encoding="utf-8")
    response = client.post(
        "/api/dataset/inputs", json={"path": dataset, "problem": str(place)}
    )
    assert response.status_code == 200, response.text
    rank = response.json()["rank"]
    assert (rank["users"], rank["items"]) == (6, 4)
    assert (rank["valid"], rank["test"]) == (None, None)
    assert rank["notes"][0] == (
        "学習の設定 (configs/default/train.yaml) を読めないので, 分ける割合"
        "は既定の値です"
    )


def test_dataset_inputs_without_rank_encoding(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    rank_encoding の無い古い ML-Quest でも, 並べる以外の問題の設定は 501
    にならず, 並べるの問題の設定だけが 501 になることを確かめる関数
    (ML-EDA と ML-Quest が無ければ飛ばす)

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        ML-Quest の rank_encoding を外す

    Returns
    ----------
    None
    """
    # 目的の列の無い表のデータセットと, 表の生成, 列の予測, 並べるの
    # 問題の設定
    ml_eda = pytest.importorskip("ml_eda")
    ml_quest = pytest.importorskip("ml_quest")
    frame = pd.DataFrame({"age": np.arange(30.0), "city": ["a", "b"] * 15})
    source = tmp_path / "table.csv"
    frame.to_csv(source, index=False)
    dataset = tmp_path / "ds"
    ml_eda.export_dataset(ml_eda.load_data(source), {
        "task": "none", "steps": [],
        "split": {"method": "random", "valid": 0.2, "test": 0.2}
    }, dataset)
    places = {}
    for name, problem in (
        ("table", table_generation_problem(ml_quest)),
        ("column", ml_quest.example_problem("総菜の作る量")),
        ("rank", rank_problem(ml_quest)),
    ):
        places[name] = tmp_path / f"{name}.yaml"
        ml_quest.save_problem(problem, places[name])

    # rank_encoding が無くても, 並べる以外はそのまま読む
    monkeypatch.delattr(ml_quest, "rank_encoding")
    for name in ("table", "column"):
        response = client.post(
            "/api/dataset/inputs",
            json={"path": str(dataset), "problem": str(places[name])}
        )
        assert response.status_code == 200, response.text
        assert response.json()["rank"] is None
    response = client.post(
        "/api/dataset/inputs",
        json={"path": str(dataset), "problem": str(places["rank"])}
    )
    assert response.status_code == 501
    assert "rank_encoding" in response.json()["detail"]
