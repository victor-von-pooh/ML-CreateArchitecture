from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from ml_modelcontrol import clear_pretrained_cache, pretrained

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
    # 一時ディレクトリを置き場にして, トークンの URL を開いた状態に設定
    app = create_app(
        data_dir=tmp_path, token="secret", allowed_hosts=["testserver"]
    )
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
    client = TestClient(create_app(
        data_dir=tmp_path, token="secret", allowed_hosts=["testserver"]
    ))
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

    # 誤りのあるノードがあっても, そこを除いた部分は形を推定
    broken = {**SPEC, "nodes": {**SPEC["nodes"], "act": {"unit": "nn.Nope"}}}
    result = client.post("/api/check", json={"spec": broken}).json()
    assert any(issue["code"] == "unknown_unit" for issue in result["issues"])
    assert result["inferred"] and not result["ok"]
    assert result["nodes"]["fc"]["outputs"]["out"]["shape"] == [4, 2]
    assert result["nodes"]["act"]["status"] == "error"

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

    # 消すと一覧から外れ, 無い定義と規則に合わない名前は消せない
    assert client.delete("/api/models/mlp").status_code == 204
    assert client.get("/api/models").json() == []
    assert client.delete("/api/models/mlp").status_code == 404
    assert client.delete("/api/models/a.b").status_code == 400

    # 消せない場合 (同じ名前のディレクトリなど) は, 無い場合と分けて 500
    (tmp_path / "models" / "dir.yaml").mkdir()
    assert client.delete("/api/models/dir").status_code == 500


def test_review_fixes(client: TestClient, tmp_path: Path) -> None:
    """
    壊れた定義のファイル, 末尾の改行の名前, 大きすぎる入力, 形の無い入力,
    型の誤りの扱いを確かめる関数

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
    # YAML として読めないファイルは 404 (500 ではない)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "bad.yaml").write_text("a: [\n", encoding="utf-8")
    assert client.get("/api/models/bad").status_code == 404

    # 末尾に改行のある名前は不可
    assert client.put(
        "/api/models/abc%0A", json={"spec": SPEC}
    ).status_code == 400

    # 大きすぎる入力の例は形を推定せずに注意
    huge = {**SPEC, "inputs": {"x": {"shape": [100000, 100000]}}}
    result = client.post("/api/check", json={"spec": huge}).json()
    assert not result["inferred"]
    assert "input_too_large" in {issue["code"] for issue in result["issues"]}

    # 形の無い入力は計算しない (推定したことにしない)
    no_shape = {**SPEC, "inputs": {"x": {}}}
    result = client.post("/api/check", json={"spec": no_shape}).json()
    assert not result["inferred"]

    # 型にリストを書いても 500 にならない
    odd = {**SPEC, "inputs": {"x": {"shape": [4, 3], "dtype": ["a"]}}}
    response = client.post("/api/check", json={"spec": odd})
    assert response.status_code == 200


def test_examples(client: TestClient) -> None:
    """
    同梱した例の一覧と読み込みを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 一覧に例があり, 読み込むと定義, 無い名前は 404
    names = {item["name"] for item in client.get("/api/examples").json()}
    assert "cnn_classifier" in names
    model = client.get("/api/examples/cnn_classifier").json()
    assert "conv1" in model["spec"]["nodes"]
    assert client.get("/api/examples/nope").status_code == 404


def test_pretrained(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    学習済みの重みのライセンスを返し, 重みを読まない場合は null を返す
    ことを確かめる関数 (ネットワークは使わない)

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    monkeypatch: pytest.MonkeyPatch
        差し替えの道具

    Returns
    ----------
    None
    """
    # Hugging Face を読む関数を, 名前ごとに決めた情報を返すものに差し替え
    clear_pretrained_cache()
    monkeypatch.setattr(pretrained, "fetch_hub_metadata", fake_hub)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    body = {"unit": "hf.AutoModel", "params": {"model_name": "org/nc"}}
    response = client.post("/api/pretrained", json=body)
    assert response.status_code == 200
    info = response.json()
    assert info["status"] == "noncommercial" and info["alert"] is True
    assert info["pinned_revision"] == "a" * 40
    assert "Hugging Face" in info["source"]

    # 重みを読まない場合は null, 登録簿に無い部品は 404
    body["params"]["pretrained"] = False
    assert client.post("/api/pretrained", json=body).json() is None
    response = client.post(
        "/api/pretrained", json={"unit": "nn.NoSuch", "params": {}}
    )
    assert response.status_code == 404
    clear_pretrained_cache()


def fake_hub(hub_id: str, revision: str) -> dict[str, Any]:
    """
    テストで Hugging Face の代わりに決めた情報を返す関数

    Parameters
    ----------
    hub_id: str
        置き場の名前
    revision: str
        版

    Returns
    ----------
    metadata: dict[str, Any]
        非商用のライセンスのモデルの情報
    """
    metadata: dict[str, Any] = {
        "sha": "a" * 40, "cardData": {"license": "cc-by-nc-4.0"}
    }

    return metadata
