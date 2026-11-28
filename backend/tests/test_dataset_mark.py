import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_createarchitecture import dataset_mark as dataset_mark_module
from ml_createarchitecture.app import create_app
from ml_createarchitecture.dataset_mark import (
    dataset_sha256,
    mark_path,
    read_mark,
)

# 全結合だけの小さなモデルの定義
SPEC: dict[str, Any] = {
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

    def model_contract(self, problem: dict[str, Any]) -> None:
        """
        骨組みの約束が無いとみなす関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定 (使わない)

        Returns
        ----------
        None
        """
        return None

    def check_model_contract(
        self, problem: dict[str, Any], spec: Any, widths: Any = None
    ) -> list[dict[str, Any]]:
        """
        約束の無い骨組みの確かめ (合わない所無し) を返す関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定 (使わない)
        spec: Any
            定義 (使わない)
        widths: Any = None
            最後の大きさの数 (使わない)

        Returns
        ----------
        found: list[dict[str, Any]]
            空の並び
        """
        found: list[dict[str, Any]] = []

        return found


def sha256_of(path: Path) -> str:
    """
    ファイルの中身の sha256 を返す関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    digest: str
        16 進の小文字
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    return digest


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
    app = create_app(
        data_dir=tmp_path / "home", token="secret",
        allowed_hosts=["testserver"]
    )
    client = TestClient(app)
    client.get("/api/auth?token=secret", follow_redirects=False)

    return client


def make_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: Path
) -> tuple[Path, Path, Path]:
    """
    学習のプロジェクトと, その記録を持つ問題の設定を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え
    dataset: Path
        記録のデータセットの場所

    Returns
    ----------
    problem: Path
        問題の設定のファイル
    project: Path
        学習のプロジェクト
    spec_path: Path
        記録の保存先
    """
    project = (tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("data:\n  dataset: outputs/dataset\n")
    problem = tmp_path / "store" / "sales.yaml"
    problem.parent.mkdir()
    problem.write_text("name: sales\n")
    spec_path = project / "models" / "model.yaml"
    record = {
        "model": {"project_model_spec": str(spec_path)},
        "data": {"project_dataset": str(dataset)},
        "project": {"path": str(project)},
    }
    monkeypatch.setitem(sys.modules, "ml_quest", StubQuest(record))

    return problem, project, spec_path


def save(
    client: TestClient, path: Path, name: Optional[str] = None
) -> dict[str, Any]:
    """
    定義を上書きで保存し, 応答を返す関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    path: Path
        問題の設定のファイルかプロジェクト
    name: Optional[str]
        models/ の下の名前

    Returns
    ----------
    saved: dict[str, Any]
        応答の本文
    """
    response = client.put("/api/project/model", json={
        "path": str(path), "name": name, "spec": SPEC, "overwrite": True
    })
    assert response.status_code == 200, response.text
    saved: dict[str, Any] = response.json()

    return saved


def test_mark_directory(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    保存の後に印のファイルがあり, データセットを書き出し直すと今の sha256
    と印が違うことと, 定義を他で書き換えた後の印と前の版の定義 (印が無い)
    を印が無い物として扱うことを確かめる関数

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
    # ディレクトリのデータセットは中の dataset.yaml の sha256 を残す
    dataset = tmp_path / "outputs" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text("format_version: 3\n")
    problem, project, spec_path = make_problem(tmp_path, monkeypatch, dataset)
    saved = save(client, problem)
    assert saved["notes"] == []
    mark = mark_path(spec_path)
    assert mark == project / "models" / "model.dataset.json"
    record = json.loads(mark.read_text())
    assert record["dataset"]["sha256"] == sha256_of(dataset / "dataset.yaml")
    assert record["spec_sha256"] == sha256_of(spec_path)
    assert not list(spec_path.parent.glob("*.tmp"))

    # 定義の中には印を入れない
    assert "dataset" not in yaml.safe_load(spec_path.read_text())
    stamp = read_mark(spec_path)
    assert stamp is not None and stamp["sha256"] == record["dataset"]["sha256"]

    # データセットを書き出し直すと, 今の sha256 と印が違う
    (dataset / "dataset.yaml").write_text("format_version: 3\nname: new\n")
    now, reason = dataset_sha256(str(dataset), project)
    assert reason is None and now != stamp["sha256"]

    # 定義を他で書き換えると, 古い印は印が無い物として扱う
    spec_path.write_text(spec_path.read_text() + "\n# 編集\n")
    assert read_mark(spec_path) is None

    # 前の版の定義 (印のファイルが無い) も印が無い物
    mark.unlink()
    assert read_mark(spec_path) is None

    # 名前で選んだ保存先も隣に印を残す
    saved = save(client, problem, "cnn")
    named = Path(saved["path"])
    assert named == project / "models" / "cnn.yaml"
    assert (project / "models" / "cnn.dataset.json").is_file()
    assert read_mark(named) is not None


def test_mark_file(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    dataset.yaml を持たないファイル 1 つのデータセットはそのファイルの
    sha256 を残すことを確かめる関数

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
    table = tmp_path / "table.csv"
    table.write_text("a,b\n1,2\n")
    problem, _, spec_path = make_problem(tmp_path, monkeypatch, table)
    save(client, problem)
    stamp = read_mark(spec_path)
    assert stamp is not None and stamp["sha256"] == sha256_of(table)


def test_mark_unreadable(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    データセットが無い, 読めない (dataset.yaml が無いディレクトリ, 末尾が
    区切りのファイル, リンクの後の ..) ときは印を残さず, 前の印を消し,
    保存は落ちないことを確かめる関数

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
    # 印のあるデータセットを, 読めない場所に替えて保存し直す
    dataset = tmp_path / "outputs" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text("format_version: 3\n")
    problem, project, spec_path = make_problem(tmp_path, monkeypatch, dataset)
    save(client, problem)
    mark = mark_path(spec_path)
    assert mark.is_file()
    table = tmp_path / "table.csv"
    table.write_text("a\n1\n")
    link = tmp_path / "link"
    link.symlink_to(dataset)
    empty = tmp_path / "empty"
    empty.mkdir()
    for value in (
        tmp_path / "missing", empty, f"{table}/", f"{link}/../outputs"
    ):
        record = sys.modules["ml_quest"].record  # type: ignore[attr-defined]
        record["data"]["project_dataset"] = str(value)
        save(client, problem)
        assert not mark.exists(), value
        assert read_mark(spec_path) is None
        # 定義は書けている
        assert spec_path.is_file()
        # 次の繰り返しのために印を戻す
        record["data"]["project_dataset"] = str(dataset)
        save(client, problem)
        assert mark.is_file()
    record = sys.modules["ml_quest"].record  # type: ignore[attr-defined]
    record["data"]["project_dataset"] = str(empty)
    saved = save(client, problem)
    assert not mark.exists() and saved["notes"] == []
    assert project.is_dir()


def test_mark_fifo(tmp_path: Path) -> None:
    """
    FIFO のデータセットで止まらず, 読めないものとして扱うことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO を作れない環境")
    fifo = tmp_path / "table.csv"
    os.mkfifo(fifo)
    assert dataset_sha256(str(fifo), tmp_path) == (None, "not_file")
    assert dataset_sha256(str(tmp_path / "x" / "y"), tmp_path) == (
        None, "missing"
    )


def test_mark_project_dir(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    学習のプロジェクトのディレクトリから決めた保存先では印を書かず, 前の
    印を消すことと, 印を書けなくても保存は落ちないことを確かめる関数

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
    dataset = tmp_path / "outputs" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text("format_version: 3\n")
    problem, project, spec_path = make_problem(tmp_path, monkeypatch, dataset)
    save(client, problem)
    mark = mark_path(spec_path)
    assert mark.is_file()

    # プロジェクトのディレクトリを選ぶと印は書かず, 前の印は消す
    (project / "outputs" / "dataset").mkdir(parents=True)
    saved = save(client, project, "model")
    assert Path(saved["path"]) == spec_path
    assert not mark.exists()

    # 印の場所がディレクトリで書けなくても, 保存は 200 で知らせを返す
    mark.mkdir()
    saved = save(client, problem)
    assert spec_path.is_file()
    assert any("印を書けません" in note for note in saved["notes"])


def test_mark_read_errors(tmp_path: Path) -> None:
    """
    印の JSON が壊れている, 64 KiB を超える, 形が違う, 版が真偽値, 深すぎる,
    印がリンクの場合に, 印が無い物として読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    spec = tmp_path / "m.yaml"
    spec.write_text("a: 1\n")
    mark = mark_path(spec)
    good = {
        "format_version": 1,
        "dataset": {"path": "d", "sha256": "0" * 64},
        "spec_sha256": sha256_of(spec),
    }
    mark.write_text(json.dumps(good))
    assert read_mark(spec) is not None

    # 読めない中身は印が無い物
    bad: list[Any] = [
        "{broken", "[" * 100000, " " * 70000 + json.dumps(good), "[]",
        json.dumps({**good, "format_version": True}),
        json.dumps({**good, "format_version": 2}),
        json.dumps({**good, "dataset": []}),
        json.dumps({**good, "spec_sha256": 1}),
        json.dumps({**good, "spec_sha256": "1" * 64}),
    ]
    for text in bad:
        mark.write_text(text)
        assert read_mark(spec) is None, text[:30]

    # 印がディレクトリか, 無いとき
    mark.unlink()
    mark.mkdir()
    assert read_mark(spec) is None
    mark.rmdir()
    assert read_mark(spec) is None


def test_hash_errors(tmp_path: Path) -> None:
    """
    読めない (権限), 循環リンク, 途中の .. の先が無い, NUL を含む値を
    求めずに理由を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 権限が無い (root は読めるので飛ばす)
    table = tmp_path / "t.csv"
    table.write_text("a\n")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        table.chmod(0)
        assert dataset_sha256(str(table), tmp_path) == (None, "unreadable")
        table.chmod(0o644)

    # 循環リンクは読めない (落ちない)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    sha, reason = dataset_sha256(str(loop), tmp_path)
    assert sha is None and reason in ("unreadable", "invalid", "missing")

    # 途中の .. の先が無い
    assert dataset_sha256(str(tmp_path / "no" / ".." / "t.csv"), tmp_path) == (
        None, "missing"
    )

    # NUL を含む値
    sha, reason = dataset_sha256("a\0b", tmp_path)
    assert sha is None and reason is not None


def test_mark_save_failures(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    保存の経路で, 一時ファイルが残らないことと, FIFO のデータセットで止まら
    ず印を書かないことと, 記録が無いとき前の印を消すことを確かめる関数

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
    dataset = tmp_path / "outputs" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text("format_version: 3\n")
    problem, _, spec_path = make_problem(tmp_path, monkeypatch, dataset)
    save(client, problem)
    mark = mark_path(spec_path)
    assert not list(spec_path.parent.glob("*.tmp"))
    assert not list(spec_path.parent.glob(".*.tmp"))
    record = sys.modules["ml_quest"].record  # type: ignore[attr-defined]

    # 書き込みが失敗しても一時ファイルは残らず, 保存は落ちない
    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("失敗")

    # (一時ファイルを作った後の書き込みで失敗させる)
    stub = type("StubJson", (), {"dumps": staticmethod(broken)})
    monkeypatch.setattr(dataset_mark_module, "json", stub)
    saved = save(client, problem)
    assert any("印を書けません" in note for note in saved["notes"])
    assert not list(spec_path.parent.glob(".*.tmp"))
    assert not mark.exists()
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "ml_quest", StubQuest(record))

    # FIFO のデータセットは止まらず, 印を書かない
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo.csv"
        os.mkfifo(fifo)
        record["data"]["project_dataset"] = str(fifo)
        save(client, problem)
        assert not mark.exists()

    # 記録が無いとき, 前の印を消す
    record["data"]["project_dataset"] = str(dataset)
    save(client, problem)
    assert mark.is_file()
    record["data"].pop("project_dataset")
    save(client, problem)
    assert not mark.exists()
