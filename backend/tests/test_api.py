from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ml_modelcontrol import clear_pretrained_cache, pretrained
from ml_modelcontrol import model as model_module

from ml_createarchitecture import api as api_module
from ml_createarchitecture.app import cookie_name, create_api, create_app

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


def test_standalone_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    単独の起動のアプリが確かめ, /api/auth, 画面の配信を持つことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        画面の置き場の差し替え

    Returns
    ----------
    None
    """
    # ビルド済みの画面の代わり
    static_dir = (tmp_path / "static").resolve()
    (static_dir / "assets").mkdir(parents=True)
    (static_dir / "index.html").write_text("画面", encoding="utf-8")
    (static_dir / "assets" / "a.js").write_text("1", encoding="utf-8")
    monkeypatch.setattr("ml_createarchitecture.app.STATIC_DIR", static_dir)
    standalone = create_app(
        data_dir=tmp_path / "home", token="secret",
        allowed_hosts=["testserver"], modules=[]
    )
    client = TestClient(standalone)

    # 違うトークンは 401, 合うトークンは Cookie を置いて画面へ
    assert client.get("/api/auth?token=x").status_code == 401
    # GET でない /api/auth もトークンの確かめの対象
    assert client.post("/api/auth?token=secret").status_code == 401
    assert client.get("/api/models").status_code == 401
    moved = client.get("/api/auth?token=secret", follow_redirects=False)
    assert moved.status_code == 303 and moved.headers["location"] == "/"
    assert cookie_name("secret") in moved.cookies
    assert client.get("/api/models").json() == []

    # ヘッダーのトークンでも呼べる
    header = TestClient(standalone).get(
        "/api/models", headers={"x-createarchitecture-token": "secret"}
    )
    assert header.status_code == 200

    # 別の Origin の書き込みは 403, 同じ Origin は受け付ける
    foreign = client.put(
        "/api/models/a", json={"spec": SPEC},
        headers={"origin": "http://evil.example"}
    )
    assert foreign.status_code == 403
    same = client.put(
        "/api/models/a", json={"spec": SPEC},
        headers={"origin": "http://testserver"}
    )
    assert same.status_code == 200, same.text
    assert (tmp_path / "home" / "models" / "a.yaml").is_file()

    # 別のホスト名は断る
    other = client.get("/api/models", headers={"host": "evil.example"})
    assert other.status_code == 400

    # 画面の配信と, 無い API と仕様の画面の 404 (根の仕様の画面も出さず,
    # 画面の中の移動先として index.html)
    assert client.get("/").text == "画面"
    assert client.get("/somewhere").text == "画面"
    assert client.get("/assets/a.js").text == "1"
    for path in ["/api", "/api/nothing", "/api/openapi.json", "/api/docs"]:
        assert client.get(path).status_code == 404, path
    for path in ["/docs", "/openapi.json"]:
        assert client.get(path).text == "画面", path


def test_create_api_mounted(tmp_path: Path) -> None:
    """
    API だけのアプリを別の頭の下に mount して, 部品の一覧, 確認, 定義の
    保存と読み込み, 問題の設定の一覧が動き, 確かめを持たないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 殻の代わりのアプリ (確かめは殻が持つので, ここでは付けない)
    problems_dir = tmp_path / "shared" / "problems"
    shell = FastAPI()
    shell.mount(
        "/api/model",
        create_api(tmp_path / "home", problems_dir=problems_dir, modules=[])
    )
    client = TestClient(shell)

    # トークン無しで, 別の Origin からも呼べる (確かめは殻の役目)
    names = {unit["name"] for unit in client.get("/api/model/units").json()}
    assert "nn.Linear" in names
    checked = client.post("/api/model/check", json={"spec": SPEC}).json()
    assert checked["ok"] and checked["params"] == 8
    saved = client.put(
        "/api/model/models/a", json={"spec": SPEC},
        headers={"origin": "http://other.example"}
    )
    assert saved.status_code == 200, saved.text
    assert (tmp_path / "home" / "models" / "a.yaml").is_file()
    listed = client.get("/api/model/models").json()
    assert [item["name"] for item in listed] == ["a"]

    # /auth も仕様の画面も持たない
    for path in ["/auth", "/openapi.json", "/docs", "/redoc"]:
        assert client.get(f"/api/model{path}").status_code == 404, path

    # 2 度呼んでも誤りにならず, 別の置き場の別のアプリ
    again = TestClient(create_api(tmp_path / "other", modules=[]))
    assert again.get("/models").json() == []


def test_create_api_problems_dir(tmp_path: Path) -> None:
    """
    問題の設定の一覧が, create_api に渡した置き場か, 渡さなければ
    ML_PATHFINDER_HOME の下を読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 渡した問題の設定の置き場に 1 つ置く (環境変数の置き場とは別)
    quest = pytest.importorskip("ml_quest")
    problems_dir = tmp_path / "shared" / "problems"
    problem = quest.example_problem(next(iter(quest.EXAMPLES)))
    problem["name"] = "試し"
    quest.save_problem(problem, problems_dir / "churn" / "problem.yaml")
    shell = FastAPI()
    shell.mount(
        "/api/model",
        create_api(tmp_path / "home", problems_dir=problems_dir, modules=[])
    )
    listed = TestClient(shell).get("/api/model/problems").json()
    assert listed["root"] == str(problems_dir.resolve())
    assert [item["name"] for item in listed["problems"]] == ["試し"]

    # problems_dir を渡さなければ ML_PATHFINDER_HOME の下の problems
    default = TestClient(create_api(tmp_path / "home", modules=[]))
    expected = (tmp_path / "pathfinder").resolve() / "problems"
    assert default.get("/problems").json()["root"] == str(expected)


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

    # 末尾に改行のある名前は不可 (単独では /api の mount の照合が改行を
    # 含む経路に合わずに 404, API だけのアプリでは名前の確かめで 400)
    assert client.put(
        "/api/models/abc%0A", json={"spec": SPEC}
    ).status_code == 404
    only_api = TestClient(create_api(tmp_path, modules=[]))
    assert only_api.put(
        "/models/abc%0A", json={"spec": SPEC}
    ).status_code == 400
    assert not list((tmp_path / "models").glob("abc*"))

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
    assert {"cnn_classifier", "tabular_autoencoder"} <= names
    model = client.get("/api/examples/cnn_classifier").json()
    assert "conv1" in model["spec"]["nodes"]
    # 異常の検知の自己符号化器は, 入力と同じ幅の再構成 recon を dec2 から出す
    model = client.get("/api/examples/tabular_autoencoder").json()
    assert model["spec"]["outputs"] == {"recon": "dec2"}
    assert model["spec"]["nodes"]["dec2"]["params"]["out_features"] == 16
    assert model["spec"]["inputs"]["x"]["shape"] == [8, 16]
    assert client.get("/api/examples/nope").status_code == 404


@pytest.fixture
def fresh_cache() -> Iterator[None]:
    """
    学習済みの重みのライセンスの覚えた結果を, テストの前後で消す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    clear_pretrained_cache()
    yield
    clear_pretrained_cache()


@pytest.mark.usefixtures("fresh_cache")
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
    monkeypatch.setattr(pretrained, "fetch_hub_metadata", fake_hub)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    body = {"unit": "hf.AutoModel", "params": {"model_name": "org/nc"}}
    response = client.post("/api/pretrained", json=body)
    assert response.status_code == 200
    info = response.json()
    assert info["status"] == "noncommercial" and info["alert"] is True
    assert info["pinned_revision"] == "a" * 40
    assert "Hugging Face" in info["source"]

    # 表示の義務, 出どころの表記の例 (作者は置き場の組織), 重みの大きさ
    assert info["attribution"]
    assert info["credit"].startswith("org の org/nc")
    assert info["author"] is None
    assert info["size_bytes"] == 1234

    # 重みを読まない場合は null, 登録簿に無い部品は 404
    body["params"]["pretrained"] = False
    assert client.post("/api/pretrained", json=body).json() is None
    response = client.post(
        "/api/pretrained", json={"unit": "nn.NoSuch", "params": {}}
    )
    assert response.status_code == 404


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
        非商用のライセンスで重みのファイルが 1 つのモデルの情報
    """
    metadata: dict[str, Any] = {
        "sha": "a" * 40, "cardData": {"license": "cc-by-nc-4.0"},
        "siblings": [{"rfilename": "model.safetensors", "size": 1234}]
    }

    return metadata


def test_weights_report(client: TestClient, tmp_path: Path) -> None:
    """
    外部の重みを読んで, ノードごとの数の報告を返すことと, 読めない場合の
    誤りの文を確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ (下の models が定義の置き場)

    Returns
    ----------
    None
    """
    # 定義の置き場に重みのファイルを置き (相対のパスの元), 一部だけを読む
    (tmp_path / "models").mkdir()
    torch.save(
        {"model.fc.weight": torch.ones(2, 3), "model.head.weight":
         torch.ones(5, 3)},
        tmp_path / "models" / "base.pt"
    )
    spec = {
        **SPEC, "format_version": 2,
        "nodes": {
            **SPEC["nodes"],
            "fc": {
                **SPEC["nodes"]["fc"],
                "train": False,
                "weights": {
                    "path": "base.pt",
                    "remap": [{"from": "model.fc.", "to": ""}],
                    "strict": False
                }
            }
        }
    }
    response = client.post("/api/weights", json={"spec": spec})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"]
    report = body["reports"][0]
    assert report["node"] == "fc"
    assert (report["loaded"], report["mismatched"], report["new"],
            report["unused"]) == (1, 0, 1, 1)
    assert report["new_keys"] == ["bias"]
    assert report["license"] is None

    # 確認は重みを読まず, 凍結を学習するパラメータ数に反映し, YAML は版 2
    checked = client.post("/api/check", json={"spec": spec}).json()
    assert checked["nodes"]["fc"]["trainable"] == 0
    assert "format_version: 2" in checked["yaml"]
    assert "format_version: 1" in client.post(
        "/api/check", json={"spec": SPEC}
    ).json()["yaml"]

    # 外部の重みが無い定義は組み立てずに空の報告
    body = client.post("/api/weights", json={"spec": SPEC}).json()
    assert body == {
        "ok": True, "reports": [], "error": None, "gated": None
    }

    # 定義の置き場の外のファイルは読まない (手元の絶対のパスも出さない)
    outside = tmp_path / "outside.pt"
    torch.save({"weight": torch.ones(2, 3)}, outside)
    away = {**spec, "nodes": {**spec["nodes"], "fc": {
        **spec["nodes"]["fc"],
        "weights": {"path": "../outside.pt", "strict": False}
    }}}
    body = client.post("/api/weights", json={"spec": away}).json()
    assert not body["ok"]
    assert "外にある" in body["error"]
    assert str(tmp_path) not in body["error"]

    # 形の推定の上限を超える大きさのモデルは組み立てない
    huge = {**spec, "nodes": {**spec["nodes"], "fc": {
        **spec["nodes"]["fc"],
        "params": {"in_features": 3, "out_features": 100_000_000}
    }}}
    body = client.post("/api/weights", json={"spec": huge}).json()
    assert not body["ok"]
    assert "上限" in body["error"]

    # strict (既定) では合わない重みで誤りの文
    del spec["nodes"]["fc"]["weights"]["strict"]
    body = client.post("/api/weights", json={"spec": spec}).json()
    assert not body["ok"]
    assert "strict: false" in body["error"]


class GatedRepoError(Exception):
    # huggingface_hub の申請制の誤りの代わり (置き場の名前を持つ)
    repo_id = "org/gated"


def test_weights_gated(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    申請制の重みを読めなかった場合は, 型の名前を付けない日本語の理由と
    配布元のページを返し, 他の誤りは型の名前を付けることを確かめる関数

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
    # トークンの無い環境で, 部品の作成が申請制の誤りで止まるように差し替え
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    def gated_build(*args: Any, **kwargs: Any) -> None:
        try:
            raise GatedRepoError("401 Client Error")
        except GatedRepoError as error:
            raise model_module.NodeBuildError("fc", error) from error

    monkeypatch.setattr(api_module, "build_model", gated_build)
    spec = {**SPEC, "format_version": 2, "nodes": {
        **SPEC["nodes"],
        "fc": {**SPEC["nodes"]["fc"], "weights": {"path": "base.pt"}}
    }}
    body = client.post("/api/weights", json={"spec": spec}).json()
    assert not body["ok"]
    assert body["error"].startswith("ノード fc の部品を作れません: 申請制")
    assert "NodeBuildError" not in body["error"]
    assert "HF_TOKEN" in body["error"]
    assert body["gated"] == {
        "reason": "no_token", "hub_id": "org/gated",
        "url": "https://huggingface.co/org/gated"
    }

    # ノードの外で起きた申請制の誤りも同じ形
    def hub_build(*args: Any, **kwargs: Any) -> None:
        raise GatedRepoError("401 Client Error")

    monkeypatch.setattr(api_module, "build_model", hub_build)
    body = client.post("/api/weights", json={"spec": spec}).json()
    assert body["error"].startswith("申請制の重み (置き場 org/gated")
    assert body["gated"]["reason"] == "no_token"

    # 申請制でない誤りは型の名前を付け, gated は null
    def other_build(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("壊れたファイル")

    monkeypatch.setattr(api_module, "build_model", other_build)
    body = client.post("/api/weights", json={"spec": spec}).json()
    assert body["error"] == "RuntimeError: 壊れたファイル"
    assert body["gated"] is None
