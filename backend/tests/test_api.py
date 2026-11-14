from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_createarchitecture.app import create_app

# 全結合と ReLU の小さなモデルの定義
SPEC = {
    "format_version": 1,
    "name": "mlp",
    "inputs": {"x": {"shape": [4, 3]}},
    "nodes": {
        "fc": {
            "unit": "nn.Linear",
            "params": {"in_features": 3, "out_features": 2},
            "inputs": "x",
            "ui": {"x": 100, "y": 40}
        },
        "act": {"unit": "nn.ReLU", "inputs": "fc"}
    },
    "outputs": {"y": "act"}
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
    # 一時ディレクトリを置き場にして, トークンの URL を開いた状態にする
    app = create_app(data_dir=tmp_path, token="secret")
    client = TestClient(app)
    client.get("/api/auth?token=secret", follow_redirects=False)

    return client


def test_requires_token(tmp_path: Path) -> None:
    """
    トークンの無い API の呼び出しと, 別のサイトからの書き込みを拒否する
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # トークンが無い場合と違う場合は 401
    client = TestClient(create_app(data_dir=tmp_path, token="secret"))
    assert client.get("/api/units").status_code == 401
    assert client.get("/api/auth?token=wrong").status_code == 401

    # 別の Origin からの書き込みは 403
    client.get("/api/auth?token=secret", follow_redirects=False)
    response = client.put(
        "/api/models/a", json={"spec": SPEC},
        headers={"origin": "http://evil.example"}
    )
    assert response.status_code == 403


def test_units_and_check(client: TestClient) -> None:
    """
    部品の一覧と, 定義の確認と形の推定を確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 部品の一覧に torch.nn の層と演算
    units = client.get("/api/units").json()
    names = {unit["name"] for unit in units}
    assert {"nn.Linear", "op.Add"} <= names

    # 正しい定義は形とパラメータ数と YAML
    result = client.post("/api/check", json={"spec": SPEC}).json()
    assert result["inferred"]
    assert result["issues"] == []
    assert result["nodes"]["fc"]["outputs"]["out"]["shape"] == [4, 2]
    assert result["params"] == 8
    assert "unit: nn.Linear" in result["yaml"]

    # 誤りのある定義は確認の結果だけ
    broken = {**SPEC, "nodes": {**SPEC["nodes"], "act": {"unit": "nn.Nope"}}}
    result = client.post("/api/check", json={"spec": broken}).json()
    assert not result["inferred"]
    assert any(issue["code"] == "unknown_unit" for issue in result["issues"])

    # 読めない定義 (新しい版) は YAML なし
    newer = {**SPEC, "format_version": 99}
    result = client.post("/api/check", json={"spec": newer}).json()
    assert result["issues"][0]["code"] == "invalid_spec"
    assert result["yaml"] == ""


def test_export(client: TestClient) -> None:
    """
    Python のコードの書き出しを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 正しい定義はコード, 誤りのある定義は 400
    response = client.post(
        "/api/export", json={"spec": SPEC, "class_name": "Mlp"}
    )
    assert "class Mlp(nn.Module):" in response.json()["code"]
    broken = {**SPEC, "outputs": {}}
    assert client.post(
        "/api/export", json={"spec": broken}
    ).status_code == 400


def test_models_save_and_load(client: TestClient, tmp_path: Path) -> None:
    """
    定義の保存, 一覧, 読み込みと, 置き場の外を指す名前の拒否を確かめる関数

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
    # 保存して一覧と読み込みで同じ定義
    assert client.get("/api/models").json() == []
    saved = client.put("/api/models/mlp", json={"spec": SPEC}).json()
    assert saved["name"] == "mlp"
    assert (tmp_path / "models" / "mlp.yaml").is_file()
    assert [item["name"] for item in client.get("/api/models").json()] == [
        "mlp"
    ]
    loaded = client.get("/api/models/mlp").json()
    assert loaded["spec"]["nodes"]["fc"]["ui"] == {"x": 100, "y": 40}

    # 置き場の外を指す名前と, 無い定義
    assert client.put(
        "/api/models/..%2Fx", json={"spec": SPEC}
    ).status_code in (400, 404, 405)
    assert client.put(
        "/api/models/a.b", json={"spec": SPEC}
    ).status_code == 400
    assert client.get("/api/models/missing").status_code == 404
