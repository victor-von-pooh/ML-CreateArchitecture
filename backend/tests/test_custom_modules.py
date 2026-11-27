import builtins
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ml_modelcontrol import register_units_from
from ml_modelcontrol.custom import LOADED_MODULES
from ml_modelcontrol.registry import REGISTRY

from ml_createarchitecture.app import create_api, create_app
from ml_createarchitecture.settings import MODULES_ENV, get_custom_modules

# 起動の設定から読み込む見本のモジュールの置き場
MODULES_DIR = Path(__file__).parent / "custom_modules"


@pytest.fixture
def lab_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """
    見本のモジュールを読み込める場所に加え, 後で登録を元に戻す関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        sys.path を一時的に変える道具

    Returns
    ----------
    path: Iterator[Path]
        見本のモジュールの置き場
    """
    monkeypatch.syspath_prepend(str(MODULES_DIR))
    yield MODULES_DIR

    # 他のテストに残さないように, 登録した部品と読み込んだ記録を消去
    for name in [name for name, unit in REGISTRY.items() if unit.module]:
        REGISTRY.pop(name)
    LOADED_MODULES.clear()
    sys.modules.pop("gui_lab_units", None)


def make_client(tmp_path: Path, modules: list[str]) -> TestClient:
    """
    自作の部品のモジュールを読み込んだテスト用のクライアントを作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    modules: list[str]
        読み込むモジュールの名前

    Returns
    ----------
    client: TestClient
        トークンを Cookie に持つクライアント
    """
    app = create_app(
        data_dir=tmp_path, token="secret", allowed_hosts=["testserver"],
        modules=modules
    )
    client = TestClient(app)
    client.get("/api/auth?token=secret", follow_redirects=False)

    return client


def test_modules_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    環境変数のモジュールの名前の読み方を確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        環境変数を一時的に変える道具

    Returns
    ----------
    None
    """
    # カンマで区切り, 空白, 空の要素, 重なりを除く
    monkeypatch.setenv(MODULES_ENV, " a.units, b ,,a.units ")
    assert get_custom_modules() == ["a.units", "b"]
    monkeypatch.delenv(MODULES_ENV)
    assert get_custom_modules() == []


def test_load_custom_modules(
    tmp_path: Path, lab_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    起動の設定のモジュールの部品が一覧に出て, 使えることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    lab_path: Path
        見本のモジュールの置き場
    capsys: pytest.CaptureFixture[str]
        標準出力を読む道具

    Returns
    ----------
    None
    """
    # 起動のログにモジュールと部品の名前
    client = make_client(tmp_path, ["gui_lab_units"])
    assert (
        "自作の部品を読み込みました: gui_lab_units → gui_lab.Shift"
        in capsys.readouterr().out
    )

    # 部品の一覧の module と, 読み込んだモジュールの一覧
    units = {unit["name"]: unit for unit in client.get("/api/units").json()}
    assert units["gui_lab.Shift"]["module"] == "gui_lab_units"
    assert units["nn.Linear"]["module"] == ""
    info = client.get("/api/modules").json()
    assert info == {
        "env": MODULES_ENV,
        "modules": [{"name": "gui_lab_units", "units": ["gui_lab.Shift"]}]
    }

    # 自作の部品を使う定義の確認と形の推定
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, 3]}},
        "nodes": {"shift": {"unit": "gui_lab.Shift", "inputs": "x"}},
        "outputs": {"y": "shift"}
    }
    result = client.post("/api/check", json={"spec": spec}).json()
    assert result["ok"]
    assert result["nodes"]["shift"]["outputs"]["out"]["shape"] == [2, 3]


def test_unknown_unit_is_not_imported(
    tmp_path: Path, lab_path: Path
) -> None:
    """
    登録されていない部品が案内付きの誤りになり, 何も読み込まないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    lab_path: Path
        見本のモジュールの置き場 (読み込める場所にあっても読み込まない)

    Returns
    ----------
    None
    """
    client = make_client(tmp_path, [])
    assert client.get("/api/modules").json()["modules"] == []

    # 読み込める場所にあるモジュールの名前を部品の名前に書いた定義
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, 3]}},
        "nodes": {
            "block": {"unit": "gui_never_import.Block", "inputs": "x"}
        },
        "outputs": {"y": "block"}
    }
    result = client.post("/api/check", json={"spec": spec}).json()
    unknown = [
        issue for issue in result["issues"]
        if issue["code"] == "unknown_unit"
    ]
    assert len(unknown) == 1
    assert "起動の設定に足してください" in unknown[0]["message"]
    response = client.post("/api/export", json={"spec": spec})
    assert response.status_code == 400

    # モジュールは読み込まれていない
    assert "gui_never_import" not in sys.modules
    assert not hasattr(builtins, "ML_CREATEARCHITECTURE_NEVER_IMPORTED")


def test_bad_module_stops_startup(tmp_path: Path, lab_path: Path) -> None:
    """
    読み込めないモジュールを書いた場合に, 起動を止めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    lab_path: Path
        見本のモジュールの置き場

    Returns
    ----------
    None
    """
    with pytest.raises(ValueError, match="no_such_gui_lab を読み込めません"):
        make_client(tmp_path, ["no_such_gui_lab"])


def test_create_api_twice_and_shared_registry(
    tmp_path: Path, lab_path: Path
) -> None:
    """
    自作の部品のモジュールを渡した create_api を 2 度呼べ, 同じプロセスの
    別のアプリが先に同じモジュールを登録していても誤りにならないことを
    確かめる関数

    殻 (ML-Studio) は 1 つのプロセスで 5 つのアプリを動かし, ML-Provide
    も ML-Inference を通して同じ ML-ModelControl の登録簿に部品を入れる

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    lab_path: Path
        見本のモジュールの置き場

    Returns
    ----------
    None
    """
    # 別のアプリの代わりに, 先に同じモジュールを登録
    assert register_units_from(["gui_lab_units"]) == {
        "gui_lab_units": ["gui_lab.Shift"]
    }

    # 同じモジュールで 2 度作っても, どちらも同じ部品を持つ
    shell = FastAPI()
    shell.mount(
        "/api/model",
        create_api(tmp_path / "a", modules=["gui_lab_units"])
    )
    shell.mount(
        "/api/again",
        create_api(tmp_path / "b", modules=["gui_lab_units"])
    )
    client = TestClient(shell)
    expected = [{"name": "gui_lab_units", "units": ["gui_lab.Shift"]}]
    for head in ["/api/model", "/api/again"]:
        assert client.get(f"{head}/modules").json()["modules"] == expected
        names = [
            unit["name"] for unit in client.get(f"{head}/units").json()
        ]
        assert names.count("gui_lab.Shift") == 1

    # モジュールを渡さないアプリは, 読み込んだモジュールを持たない
    # (登録簿の部品は同じプロセスの全てのアプリから見える)
    empty = TestClient(create_api(tmp_path / "c", modules=[]))
    assert empty.get("/modules").json()["modules"] == []
