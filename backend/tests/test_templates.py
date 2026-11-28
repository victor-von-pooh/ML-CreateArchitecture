import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ml_modelcontrol import load_example
from ml_modelcontrol.templates import TEMPLATE_ARGS

from ml_createarchitecture.app import create_app


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


@pytest.mark.parametrize("name", list(TEMPLATE_ARGS))
def test_match_bundled(client: TestClient, name: str) -> None:
    """
    同梱の雛形のままの定義を, その雛形と引数の今の値として見分けることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    name: str
        雛形の名前

    Returns
    ----------
    None
    """
    # 雛形の名前と, 全ての引数と場所
    response = client.post(
        "/api/templates/match", json={"spec": load_example(name)}
    )
    assert response.status_code == 200
    match = response.json()
    assert match["name"] == name
    assert [arg["name"] for arg in match["args"]] == list(TEMPLATE_ARGS[name])
    for arg in match["args"]:
        assert len(arg["places"]) == len(TEMPLATE_ARGS[name][arg["name"]])


def test_match_ignores_layout(client: TestClient) -> None:
    """
    位置, 名前, 説明, 入力の形と型を変えても雛形とみなし, 一度書き換えた
    雛形も今の値で見分けることを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 位置, 名前, 入力の形と型を変え, 入力の幅と水準の数を書き換えた定義
    spec = load_example("tabular_demand")
    spec["name"] = "mine"
    spec.pop("description")
    spec["nodes"]["demand1"]["ui"] = {"x": 1, "y": 2}
    spec["inputs"]["x"] = {"shape": [4, 5], "dtype": "float32"}
    spec["nodes"]["demand1"]["params"]["in_features"] = 5
    spec["nodes"]["demand2"]["params"]["out_features"] = 4
    response = client.post("/api/templates/match", json={"spec": spec})
    match = response.json()
    assert match["name"] == "tabular_demand"
    assert {arg["name"]: arg["value"] for arg in match["args"]} == {
        "features": 5, "levels": 4
    }


@pytest.mark.parametrize("change", ["param", "node", "wire", "missing"])
def test_match_changed(client: TestClient, change: str) -> None:
    """
    引数, 部品, つなぎ方を変えた定義と, 同梱の例でない定義を雛形と
    みなさないことを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    change: str
        変え方

    Returns
    ----------
    None
    """
    # 1 か所だけ変えた雛形
    spec = load_example("tabular_autoencoder")
    if change == "param":
        spec["nodes"]["enc1"]["params"]["out_features"] = 7
    elif change == "node":
        spec["nodes"]["extra"] = {"unit": "nn.ReLU", "inputs": "enc1"}
    elif change == "wire":
        spec["nodes"]["dec2"]["inputs"] = "enc1"
    else:
        spec = load_example("mlp")
    response = client.post("/api/templates/match", json={"spec": spec})
    assert response.status_code == 200
    assert response.json() is None


def test_adapt_keeps_layout(client: TestClient) -> None:
    """
    渡した引数を書き換え, 位置, 名前, 入力の書き換えない軸と型を今の定義
    のまま残すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 位置と名前と入力の高さと幅を変えた U-Net
    spec = load_example("unet_segmentation")
    spec["name"] = "mine"
    spec["nodes"]["head"]["ui"] = {"x": 5, "y": 6}
    spec["inputs"]["image"]["shape"] = [2, 3, 64, 64]
    spec["inputs"]["image"]["dtype"] = "float32"
    response = client.post(
        "/api/templates/adapt",
        json={
            "spec": spec, "name": "unet_segmentation",
            "values": {"channels": 1, "classes": 4}
        }
    )
    assert response.status_code == 200
    adapted = response.json()
    result = adapted["spec"]
    assert result["name"] == "mine"
    assert result["nodes"]["head"]["ui"] == {"x": 5, "y": 6}
    assert result["nodes"]["head"]["params"]["out_channels"] == 4
    assert result["nodes"]["enc1a"]["params"]["in_channels"] == 1
    assert result["inputs"]["image"]["shape"] == [2, 1, 64, 64]
    assert result["inputs"]["image"]["dtype"] == "float32"
    assert [
        (arg["name"], arg["before"], arg["after"]) for arg in adapted["args"]
    ] == [("channels", 3, 1), ("classes", 2, 4)]


def test_adapt_keeps_current_values(client: TestClient) -> None:
    """
    渡さない引数は, 同梱の例でなく今の定義の値のまま残し, 倍率を
    かけた場所もそろえることを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 出す点の数を書き換えた後に, 入力の点の数だけを渡す
    first = client.post(
        "/api/templates/adapt",
        json={
            "spec": load_example("point_cloud_predictor"),
            "name": "point_cloud_predictor", "values": {"points_out": 100}
        }
    ).json()["spec"]
    assert first["nodes"]["decode2"]["params"]["out_features"] == 300
    assert first["nodes"]["cloud"]["params"]["shape"][0] == 100
    second = client.post(
        "/api/templates/adapt",
        json={
            "spec": first, "name": "point_cloud_predictor",
            "values": {"points_in": 64}
        }
    ).json()
    assert second["spec"]["nodes"]["cloud"]["params"]["shape"][0] == 100
    assert second["spec"]["inputs"]["observed"]["shape"][1] == 64
    assert [arg["name"] for arg in second["args"]] == ["points_in"]


def test_adapt_errors(client: TestClient) -> None:
    """
    書き換えられない雛形は 404, 雛形のままでない定義は 409, 値の誤りは
    422 を返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント

    Returns
    ----------
    None
    """
    # 雛形でない名前, 変えた定義, 知らない引数と 1 未満と真偽値の値
    spec = load_example("tabular_price")
    response = client.post(
        "/api/templates/adapt",
        json={"spec": spec, "name": "mlp", "values": {}}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "書き換えられる雛形 mlp が見つかりません"
    changed = copy.deepcopy(spec)
    changed["nodes"]["units1"]["params"]["out_features"] = 3
    response = client.post(
        "/api/templates/adapt",
        json={
            "spec": changed, "name": "tabular_price",
            "values": {"features": 3}
        }
    )
    assert response.status_code == 409
    for values in ({"levels": 3}, {"features": 0}, {"features": True}):
        response = client.post(
            "/api/templates/adapt",
            json={"spec": spec, "name": "tabular_price", "values": values}
        )
        assert response.status_code == 422
