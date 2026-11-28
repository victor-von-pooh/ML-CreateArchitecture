import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_createarchitecture.app import create_app
from ml_createarchitecture.dataset_mark import mark_path, write_mark

IDENT = "0123456789ab"


class StubQuest:
    def __init__(self, record: dict[str, Any]):
        # handoff が返す値を保存
        self.record = record
        # 書きかけの判定 (ml.use の答えと, 確かめの結果)
        self.use: Any = True
        self.items: list[dict[str, Any]] = []

    def upgrade_problem(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        問題の設定をそのまま返す関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定

        Returns
        ----------
        problem: dict[str, Any]
            同じ値
        """
        return problem

    def get_value(self, problem: dict[str, Any], key: str) -> Any:
        """
        ml.use の答えを返す関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定 (使わない)
        key: str
            鍵 (使わない)

        Returns
        ----------
        value: Any
            保存した答え
        """
        return self.use

    def check(self, problem: dict[str, Any]) -> list[dict[str, Any]]:
        """
        保存した確かめの結果を返す関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定 (使わない)

        Returns
        ----------
        items: list[dict[str, Any]]
            確かめの結果
        """
        return self.items

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

    def stage_needs(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        問題の設定の needs をそのまま段階の要否として返す関数

        Parameters
        ----------
        problem: dict[str, Any]
            問題の設定 (model と data の要否を持つ)

        Returns
        ----------
        needs: dict[str, Any]
            段階ごとの要否 (書かない段階は要る)
        """
        given = problem["needs"]
        needs = {
            key: {"needed": given.get(key, True)} for key in ("model", "data")
        }

        return needs


class Setup:
    def __init__(
        self, client: TestClient, record: dict[str, Any], project: Path,
        dataset: Path, spec: Path, problem: Path
    ):
        # 試験の場の物を保存
        self.client = client
        self.record = record
        self.project = project
        self.dataset = dataset
        self.spec = spec
        self.problem = problem

    def progress(self) -> dict[str, Any]:
        """
        進みを引く関数

        Returns
        ----------
        progress: dict[str, Any]
            応答の JSON
        """
        response = self.client.get("/api/progress", params={"problem": IDENT})
        assert response.status_code == 200, response.text
        progress: dict[str, Any] = response.json()

        return progress


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


@pytest.fixture
def setup(
    client: TestClient, tmp_path: Path, pathfinder_home: Path,
    monkeypatch: pytest.MonkeyPatch
) -> Setup:
    """
    記録が全て揃った問題 (定義は未保存) を作る関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    tmp_path: Path
        一時ディレクトリ
    pathfinder_home: Path
        問題の置き場の元
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え

    Returns
    ----------
    setup: Setup
        試験の場
    """
    project = (tmp_path / "proj").resolve()
    config = project / "configs" / "default" / "train.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model:\n  spec: models/model.yaml\n")
    dataset = project / "outputs" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "dataset.yaml").write_text("name: a\n")
    spec = project / "models" / "model.yaml"
    record = {
        "model": {"project_model_spec": str(spec)},
        "data": {"project_dataset": str(dataset)},
        "project": {"path": str(project)},
    }
    problem = pathfinder_home / "problems" / IDENT / "problem.yaml"
    problem.parent.mkdir(parents=True)
    problem.write_text("needs: {}\n")
    monkeypatch.setitem(sys.modules, "ml_quest", StubQuest(record))

    return Setup(client, record, project, dataset, spec, problem)


def save_spec(setup: Setup, mark: bool = True) -> None:
    """
    定義と印を保存する関数

    Parameters
    ----------
    setup: Setup
        試験の場
    mark: bool
        印も残すか

    Returns
    ----------
    None
    """
    setup.spec.parent.mkdir(exist_ok=True)
    setup.spec.write_text("name: m\n")
    if mark:
        write_mark(setup.spec, str(setup.dataset), setup.project)


def test_none(setup: Setup) -> None:
    """
    前提があり定義が無いと none になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    progress = setup.progress()
    assert progress["state"] == "none" and progress["back"] is None

    # models/ の下の別の名前の定義は数えない
    setup.spec.parent.mkdir()
    (setup.spec.parent / "other.yaml").write_text("name: o\n")
    assert setup.progress()["state"] == "none"


def test_done_and_stale(setup: Setup) -> None:
    """
    印が今のデータセットと同じなら done, 違えば stale になり,
    印が無い定義は done で新しさ不明と書くことを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    save_spec(setup)
    progress = setup.progress()
    assert progress["state"] == "done" and progress["back"] is None
    assert "分かりません" not in progress["summary"]

    # データセットを書き換えると古い (GET は印を動かさない)
    (setup.dataset / "dataset.yaml").write_text("name: b\n")
    progress = setup.progress()
    assert progress["state"] == "stale" and progress["back"] == "data"
    assert setup.spec.read_text() == "name: m\n"

    # 印が無ければ新しさ不明の done
    mark_path(setup.spec).unlink()
    progress = setup.progress()
    assert progress["state"] == "done"
    assert "分かりません" in progress["summary"]


def test_blocked(setup: Setup) -> None:
    """
    前提が足りない各条件が blocked と戻る段階になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    # データセットの記録が無いか, 記録の場所に無い
    dataset = setup.record["data"].pop("project_dataset")
    assert setup.progress()["back"] == "data"
    setup.record["data"]["project_dataset"] = str(setup.project / "none")
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "data"
    setup.record["data"]["project_dataset"] = dataset

    # 学習のプロジェクトの記録の場所に config が無い
    (setup.project / "configs" / "default" / "train.yaml").unlink()
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"

    # 学習のプロジェクトの記録が無い (データセットより先に採る)
    setup.record["data"].pop("project_dataset")
    del setup.record["project"]["path"]
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"


def test_skip(setup: Setup) -> None:
    """
    要らない段階は skip, データが要らない問題はデータセットを前提に
    しないことを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    # データセットの記録が無くても, data が要らなければ前提に数えない
    setup.record["data"].clear()
    setup.problem.write_text("needs: {data: false}\n")
    assert setup.progress()["state"] == "none"
    save_spec(setup, mark=False)
    progress = setup.progress()
    assert progress["state"] == "done"
    assert "分かりません" not in progress["summary"]

    # model が要らなければ, 前提が足りなくても skip
    setup.problem.write_text("needs: {model: false}\n")
    del setup.record["project"]["path"]
    progress = setup.progress()
    assert progress["state"] == "skip" and progress["back"] is None


@pytest.mark.parametrize("ident", ["abc", "0123456789AB", "../" + IDENT])
def test_not_found_shape(setup: Setup, ident: str) -> None:
    """
    形の違う識別子と無い問題は 404 になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    ident: str
        識別子

    Returns
    ----------
    None
    """
    response = setup.client.get("/api/progress", params={"problem": ident})
    assert response.status_code == 404
    response = setup.client.get(
        "/api/progress", params={"problem": "ffffffffffff"}
    )
    assert response.status_code == 404


def test_not_found_link(setup: Setup, tmp_path: Path) -> None:
    """
    problem.yaml がリンクなら 404 になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    real = tmp_path / "real.yaml"
    real.write_text("needs: {}\n")
    setup.problem.unlink()
    setup.problem.symlink_to(real)
    response = setup.client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 404


def test_broken_problem(setup: Setup) -> None:
    """
    壊れた問題は 422 になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    setup.problem.write_text("a: [1\n")
    response = setup.client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 422


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root は読める"
)
def test_unreadable(
    setup: Setup, caplog: pytest.LogCaptureFixture
) -> None:
    """
    読めない場所は 422 で理由の文だけを返し, 警告を残すことを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    caplog: pytest.LogCaptureFixture
        ログの取得

    Returns
    ----------
    None
    """
    save_spec(setup)
    setup.project.chmod(0)
    try:
        response = setup.client.get(
            "/api/progress", params={"problem": IDENT}
        )
    finally:
        setup.project.chmod(0o755)
    assert response.status_code == 422
    assert str(setup.project) not in response.text
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_shape_error_is_500(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    記録の形の誤り (KeyError など) は握らず 500 になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え

    Returns
    ----------
    None
    """
    stub = StubQuest({})
    monkeypatch.setattr(stub, "stage_needs", lambda problem: {})
    monkeypatch.setitem(sys.modules, "ml_quest", stub)
    client = TestClient(setup.client.app, raise_server_exceptions=False)
    client.get("/api/auth?token=secret", follow_redirects=False)
    response = client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 500


def test_unfinished_problem(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    書きかけの問題 (ml.use が真でない, error が残る) は blocked になり,
    他の handoff の ValueError は 500, 版を読めない問題は固定の文の 422
    になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    monkeypatch: pytest.MonkeyPatch
        モジュールの差し替え

    Returns
    ----------
    None
    """
    stub = StubQuest(setup.record)

    def unfinished(problem: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("答えていません")

    monkeypatch.setitem(sys.modules, "ml_quest", stub)

    # 機械学習を使うか答えていない
    stub.use = None
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"

    # 確かめに error が残る
    stub.use = True
    stub.items = [{"level": "error"}]
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"

    # 他の ValueError (形の誤り) は握らない
    stub.items = [{"level": "warning"}]
    monkeypatch.setattr(stub, "handoff", unfinished)
    quiet = TestClient(setup.client.app, raise_server_exceptions=False)
    quiet.get("/api/auth?token=secret", follow_redirects=False)
    response = quiet.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 500

    # 版を読めない問題は例外の文を出さない
    monkeypatch.setattr(stub, "stage_needs", unfinished)
    response = setup.client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 422
    assert "答えていません" not in response.text


def test_broken_problem_message(setup: Setup) -> None:
    """
    壊れた問題の 422 は例外の文とパスを含まないことを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    setup.problem.write_text("a: [1\n")
    response = setup.client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 422
    assert "yaml" not in response.text.lower()
    assert str(setup.problem) not in response.text


def test_shape_error_section(setup: Setup) -> None:
    """
    辞書でない段階の値は握らず 500 になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    setup.record["model"] = []
    client = TestClient(setup.client.app, raise_server_exceptions=False)
    client.get("/api/auth?token=secret", follow_redirects=False)
    response = client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 500


def test_spec_outside_or_relative(setup: Setup, tmp_path: Path) -> None:
    """
    保存先が学習のプロジェクトの外 (.. とリンク) か相対なら
    blocked になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 外の場所 (.. を含む書き方)
    outside = tmp_path / "outside"
    outside.mkdir()
    setup.record["model"]["project_model_spec"] = str(
        setup.project / ".." / "outside" / "m.yaml"
    )
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"

    # プロジェクトの中のリンクが外を指す
    (setup.project / "models").mkdir()
    (setup.project / "models" / "out").symlink_to(outside)
    setup.record["model"]["project_model_spec"] = str(
        setup.project / "models" / "out" / "m.yaml"
    )
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"

    # 相対の記録
    setup.record["model"]["project_model_spec"] = "models/model.yaml"
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"
    setup.record["project"]["path"] = "proj"
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"


def test_config_is_directory(setup: Setup) -> None:
    """
    train.yaml がディレクトリなら blocked になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    config = setup.project / "configs" / "default" / "train.yaml"
    config.unlink()
    config.mkdir()
    progress = setup.progress()
    assert progress["state"] == "blocked" and progress["back"] == "problem"


def test_place_link_outside(setup: Setup, tmp_path: Path) -> None:
    """
    置き場の外を指す問題のディレクトリのリンクは 404 になることを
    確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    other = tmp_path / "other"
    other.mkdir()
    (other / "problem.yaml").write_text("needs: {}\n")
    link = setup.problem.parent.parent / "abcdefabcdef"
    link.symlink_to(other)
    response = setup.client.get(
        "/api/progress", params={"problem": "abcdefabcdef"}
    )
    assert response.status_code == 404


def test_mark_of_other_dataset(setup: Setup, tmp_path: Path) -> None:
    """
    印が別のデータセットを指すなら新しさ不明の done になることを
    確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    other = tmp_path / "other"
    other.mkdir()
    (other / "dataset.yaml").write_text("name: z\n")
    save_spec(setup)
    setup.record["data"]["project_dataset"] = str(other)
    progress = setup.progress()
    assert progress["state"] == "done"
    assert "分かりません" in progress["summary"]


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root は読める"
)
def test_unreadable_dataset(setup: Setup) -> None:
    """
    読めないデータセットは 422 になることを確かめる関数

    Parameters
    ----------
    setup: Setup
        試験の場

    Returns
    ----------
    None
    """
    setup.dataset.chmod(0)
    try:
        response = setup.client.get(
            "/api/progress", params={"problem": IDENT}
        )
    finally:
        setup.dataset.chmod(0o755)
    assert response.status_code == 422


def real_problem(
    setup_home: Path, change: Any = None
) -> None:
    """
    実物の ML-Quest の例の問題を置き場に書く関数

    Parameters
    ----------
    setup_home: Path
        問題の置き場の元
    change: Any
        問題の設定を書き換える関数 (無ければそのまま)

    Returns
    ----------
    None
    """
    quest = pytest.importorskip("ml_quest")
    problem = quest.example_problem(next(iter(quest.EXAMPLES)))
    if change is not None:
        change(problem)
    path = setup_home / "problems" / IDENT / "problem.yaml"
    quest.save_problem(problem, path)


def test_real_quest_skip(client: TestClient, pathfinder_home: Path) -> None:
    """
    実物の ML-Quest で, 機械学習を使わない問題が skip になることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    pathfinder_home: Path
        問題の置き場の元

    Returns
    ----------
    None
    """
    def change(problem: dict[str, Any]) -> None:
        problem["ml"] = {"use": False, "reason": "表で足りる"}

    real_problem(pathfinder_home, change)
    response = client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "skip"


def test_real_quest_blocked(client: TestClient, pathfinder_home: Path) -> None:
    """
    実物の ML-Quest で, 何も書き出していない問題が blocked になることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        API を呼ぶクライアント
    pathfinder_home: Path
        問題の置き場の元

    Returns
    ----------
    None
    """
    real_problem(pathfinder_home)
    response = client.get("/api/progress", params={"problem": IDENT})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "blocked" and body["back"] == "problem"
