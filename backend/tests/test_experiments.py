import hashlib
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from fastapi.testclient import TestClient

from ml_createarchitecture.app import create_app

# 学習に使った定義 (版 1)
SPEC: dict[str, Any] = {
    "format_version": 1,
    "name": "mlp",
    "inputs": {"x": {"shape": [2, 4]}},
    "nodes": {
        "fc": {
            "unit": "nn.Linear",
            "params": {"in_features": 4, "out_features": 2},
            "inputs": "x",
        }
    },
    "outputs": {"y": "fc"},
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
    app = create_app(
        data_dir=tmp_path / "home", token="secret",
        allowed_hosts=["testserver"]
    )
    client = TestClient(app)
    client.get("/api/auth?token=secret", follow_redirects=False)

    return client


def make_project(path: Path, output_dir: Optional[str] = None) -> Path:
    """
    学習のプロジェクトの形のディレクトリを作る関数

    Parameters
    ----------
    path: Path
        プロジェクトのディレクトリ
    output_dir: Optional[str]
        config の experiment.output_dir (None なら書かない)

    Returns
    ----------
    path: Path
        プロジェクトのディレクトリ
    """
    config = path / "configs" / "default" / "train.yaml"
    config.parent.mkdir(parents=True)
    text = "model:\n  spec: models/model.yaml\n"
    if output_dir is not None:
        text += f"experiment:\n  output_dir: {output_dir}\n"
    config.write_text(text)

    return path


def make_run(
    root: Path, run: str, spec: Optional[dict[str, Any]] = None,
    source: str = "models/mlp.yaml", status: str = "copied",
    sha: Optional[str] = None
) -> Path:
    """
    ML-Found の形の実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        実験の置き場
    run: str
        <名前>/<日付>/<時刻>
    spec: Optional[dict[str, Any]]
        写した定義 (None なら SPEC)
    source: str
        source.yaml の model_spec.path
    status: str
        source.yaml の model_spec.status
    sha: Optional[str]
        記録する sha256 (None なら写しの中身のもの)

    Returns
    ----------
    run_dir: Path
        実験のディレクトリ
    """
    run_dir = root / run
    run_dir.mkdir(parents=True)
    data = yaml.safe_dump(spec or SPEC).encode()
    (run_dir / "model_spec.yaml").write_bytes(data)
    (run_dir / "config.yaml").write_text("model:\n  spec: x\n")
    record = {
        "path": source, "status": status,
        "sha256": sha or hashlib.sha256(data).hexdigest(),
    }
    (run_dir / "source.yaml").write_text(
        yaml.safe_dump({"format_version": 1, "model_spec": record})
    )
    (run_dir / "run.json").write_text('{"status": "completed"}')

    return run_dir


def test_list_and_open(client: TestClient, tmp_path: Path) -> None:
    # 既定の outputs の下の実験を並べ, 写しを開ける
    project = make_project(tmp_path / "proj")
    make_run(project / "outputs", "mlp/2026-11-20/10-00-00")
    response = client.post(
        "/api/project/experiments", json={"path": str(project)}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["project"] == str(project.resolve())
    [entry] = body["experiments"]
    assert entry["run"] == "mlp/2026-11-20/10-00-00"
    assert entry["usable"] is True
    assert entry["name"] == "mlp"
    assert entry["status"] == "completed"
    assert entry["spec"] == "models/mlp.yaml"

    opened = client.post(
        "/api/project/experiment-spec",
        json={"path": str(project), "run": entry["run"]},
    )
    assert opened.status_code == 200
    assert opened.json()["name"] == "mlp"
    assert opened.json()["spec"]["nodes"]["fc"]["unit"] == "nn.Linear"


def test_output_dir_from_config(client: TestClient, tmp_path: Path) -> None:
    # config の experiment.output_dir の下を探す
    project = make_project(tmp_path / "proj", output_dir="runs")
    make_run(project / "runs", "a/d/t")
    make_run(project / "outputs", "b/d/t")
    body = client.post(
        "/api/project/experiments", json={"path": str(project)}
    ).json()
    assert [item["run"] for item in body["experiments"]] == ["a/d/t"]


def test_unusable_copies(client: TestClient, tmp_path: Path) -> None:
    # 記録の無い写し, 中身が記録と違う写し, リンクの写しは開けない
    project = make_project(tmp_path / "proj")
    root = project / "outputs"
    make_run(root, "a/d/1", status="missing")
    make_run(root, "a/d/2", sha="0" * 64)
    linked = make_run(root, "a/d/3")
    outside = tmp_path / "outside.yaml"
    outside.write_text(yaml.safe_dump(SPEC))
    (linked / "model_spec.yaml").unlink()
    (linked / "model_spec.yaml").symlink_to(outside)
    body = client.post(
        "/api/project/experiments", json={"path": str(project)}
    ).json()
    reasons = {item["run"]: item for item in body["experiments"]}
    assert all(not item["usable"] for item in reasons.values())
    assert "記録がありません" in reasons["a/d/1"]["reason"]
    assert "違います" in reasons["a/d/2"]["reason"]
    assert "普通のファイル" in reasons["a/d/3"]["reason"]
    for run in reasons:
        response = client.post(
            "/api/project/experiment-spec",
            json={"path": str(project), "run": run},
        )
        assert response.status_code == 422


@pytest.mark.parametrize(
    "run", ["../../x/d/t", "a/d", "/etc/passwd", "a/d/t/../../../.."]
)
def test_open_only_listed(
    client: TestClient, tmp_path: Path, run: str
) -> None:
    # 一覧に無い実験は, 送られた文字から場所を組み立てずに 404
    project = make_project(tmp_path / "proj")
    make_run(project / "outputs", "a/d/t")
    make_run(tmp_path / "x", "d/t/u")
    response = client.post(
        "/api/project/experiment-spec",
        json={"path": str(project), "run": run},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("output_dir", ["../elsewhere", "/tmp", "."])
def test_output_dir_outside(
    client: TestClient, tmp_path: Path, output_dir: str
) -> None:
    # プロジェクトの外とプロジェクトそのものの置き場は読まない
    project = make_project(tmp_path / "proj", output_dir=output_dir)
    make_run(tmp_path / "elsewhere", "a/d/t")
    body = client.post(
        "/api/project/experiments", json={"path": str(project)}
    ).json()
    assert body["experiments"] == []
    assert body["roots"] == []
    assert body["notes"]


def test_linked_dirs_skipped(client: TestClient, tmp_path: Path) -> None:
    # 置き場と途中のディレクトリのリンクはたどらない
    outside = tmp_path / "outside"
    make_run(outside, "a/d/t")
    linked = make_project(tmp_path / "linked")
    (linked / "outputs").symlink_to(outside)
    body = client.post(
        "/api/project/experiments", json={"path": str(linked)}
    ).json()
    assert body["experiments"] == []
    project = make_project(tmp_path / "proj")
    (project / "outputs").mkdir()
    (project / "outputs" / "a").symlink_to(outside / "a")
    body = client.post(
        "/api/project/experiments", json={"path": str(project)}
    ).json()
    assert body["experiments"] == []


def test_name_from_relaunched_copy(
    client: TestClient, tmp_path: Path
) -> None:
    # 写しから起動し直した実験の名前の候補は, 実験の名前
    project = make_project(tmp_path / "proj")
    make_run(
        project / "outputs", "cnn/d/t",
        source="outputs/cnn/d/s/model_spec.yaml"
    )
    [entry] = client.post(
        "/api/project/experiments", json={"path": str(project)}
    ).json()["experiments"]
    assert entry["name"] == "cnn"


def test_not_project(client: TestClient, tmp_path: Path) -> None:
    # 学習のプロジェクトでない場所と, 無い場所
    (tmp_path / "plain").mkdir()
    response = client.post(
        "/api/project/experiments", json={"path": str(tmp_path / "plain")}
    )
    assert response.status_code == 422
    response = client.post(
        "/api/project/experiments", json={"path": str(tmp_path / "nope")}
    )
    assert response.status_code == 404


def test_open_keeps_saved(client: TestClient, tmp_path: Path) -> None:
    # 開いた写しを models/ に保存しても, 既にある定義は確かめずに上書き
    # しない (今の保存の守りのまま)
    project = make_project(tmp_path / "proj")
    make_run(project / "outputs", "mlp/d/t")
    (project / "models").mkdir()
    (project / "models" / "mlp.yaml").write_text("keep\n")
    opened = client.post(
        "/api/project/experiment-spec",
        json={"path": str(project), "run": "mlp/d/t"},
    ).json()
    response = client.put(
        "/api/project/model",
        json={"path": str(project), "name": opened["name"],
              "spec": opened["spec"]},
    )
    assert response.status_code == 409
    assert (project / "models" / "mlp.yaml").read_text() == "keep\n"
