"""
学習の実験に残ったモデルの定義の写しを開く API

ML-Found は実験のディレクトリ (<output_dir>/<名前>/<日付>/<時刻>/) に, 学習
に使った定義を model_spec.yaml として写し, 中身の sha256 を source.yaml の
model_spec に書く. 候補を比べて選んだ定義を, 名前を付けてプロジェクトの
models/ に保存し直すために, その写しを画面に開く. 読むのはプロジェクトの
中の実験の置き場だけで, リンクはたどらない
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException
from ml_modelcontrol import load_spec
from pydantic import BaseModel, Field

from ml_createarchitecture.project import (
    PROBLEM_FILE,
    PROJECT_CONFIG,
    PathBody,
    absolute_path,
    from_problem,
    missing_record,
    modified_at,
    section,
)
from ml_createarchitecture.store import NAME_PATTERN

router = APIRouter(prefix="/api")

# 実験の置き場を書く config (ML-Dashboard が実験を探す config と同じ) と,
# 書いていない場合の ML-Found の既定の出力先
OUTPUT_CONFIGS = (
    Path("configs/default/train.yaml"),
    Path("configs/experiment/train.yaml"),
)
DEFAULT_OUTPUT = "outputs"

# 実験のディレクトリのファイル (ML-Found の docs/FORMAT.md)
SPEC_COPY = "model_spec.yaml"
RUN_FILES = ("config.yaml", "source.yaml")

# 読む写しの大きさの上限 (ML-Found が写す上限と同じ 16 MiB)
MAX_SPEC_BYTES = 16 * 1024 * 1024


class ExperimentBody(BaseModel):
    path: str = Field(
        description="プロジェクトのディレクトリか問題の設定のファイルの絶対"
        "のパス (~ から書ける)"
    )
    run: str = Field(
        description="実験の置き場からの実験のディレクトリ (<名前>/<日付>/"
        "<時刻>. 一覧の run)"
    )


class ExperimentEntry(BaseModel):
    # 実験と, その定義の写しを開けるか
    run: str = Field(description="実験の置き場からの <名前>/<日付>/<時刻>")
    path: str = Field(description="実験のディレクトリ")
    spec: Optional[str] = Field(
        default=None,
        description="写した元の定義の場所 (source.yaml の model_spec.path)"
    )
    name: str = Field(description="保存する名前の候補")
    status: Optional[str] = Field(
        default=None, description="run.json の状態"
    )
    updated_at: Optional[str] = None
    usable: bool
    reason: Optional[str] = None


class ExperimentList(BaseModel):
    # プロジェクト, 探した実験の置き場, 新しい順の実験, 読めなかった理由
    project: str
    roots: list[str]
    experiments: list[ExperimentEntry]
    notes: list[str] = Field(default_factory=list)


class ExperimentSpec(BaseModel):
    # 開く定義, 保存する名前の候補, 実験
    name: str
    spec: dict[str, Any]
    run: str
    path: str


def project_dir(text: str) -> Path:
    """
    プロジェクトのディレクトリか問題の設定のファイルから, プロジェクトの
    ディレクトリを求める関数

    Parameters
    ----------
    text: str
        プロジェクトのディレクトリか問題の設定のファイル

    Returns
    ----------
    project: Path
        学習のプロジェクトのディレクトリ

    Raises
    ----------
    HTTPException
        場所が無い場合は 404, 学習のプロジェクトを決められない場合は 422,
        ML-Quest が要るのに無い場合は 501
    """
    # 問題の設定は記録のプロジェクト (保存先と同じ決め方)
    path = absolute_path(text)
    if path.is_dir() and (path / PROJECT_CONFIG).is_file():
        project = path
        return project
    problem = path if path.is_file() else path / PROBLEM_FILE
    if not path.exists():
        raise HTTPException(404, f"{path} がありません")
    if not problem.is_file():
        raise HTTPException(
            422,
            f"{path} は学習のプロジェクト ({PROJECT_CONFIG} を持つ"
            "ディレクトリ) ではありません. プロジェクトのディレクトリか"
            "問題の設定のファイルを選んでください"
        )
    found = from_problem(problem)
    if found is None or found.project is None:
        raise HTTPException(
            422, f"{problem.name} を選べません: {missing_record(problem)}"
        )
    project = Path(found.project)

    return project


def output_roots(project: Path) -> tuple[list[Path], list[str]]:
    """
    config の experiment.output_dir から実験の置き場を求める関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    roots: list[Path]
        プロジェクトの中の実験の置き場 (重なりは 1 つに)
    notes: list[str]
        読まない置き場の理由
    """
    # 書いていない config と読めない config は既定の outputs
    values: list[str] = []
    for name in OUTPUT_CONFIGS:
        file = project / name
        if not file.is_file():
            continue
        try:
            config = yaml.safe_load(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            config = None
        value = section(config, "experiment").get("output_dir")
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    if not values:
        values.append(DEFAULT_OUTPUT)

    # プロジェクトの中で, 途中にリンクの無い置き場だけを読む (決まった
    # 場所の外を読まないため)
    roots: list[Path] = []
    notes: list[str] = []
    for value in values:
        if "${" in value or "://" in value:
            notes.append(f"実験の置き場 {value} は参照か URL なので読みません")
            continue
        raw = Path(os.path.normpath(project / Path(value).expanduser()))
        try:
            root = raw.resolve()
        except (OSError, RuntimeError):
            continue
        if not root.is_relative_to(project) or root == project:
            notes.append(
                f"実験の置き場 {value} がプロジェクトの外か, プロジェクト"
                "そのものなので読みません"
            )
            continue
        if root != raw:
            notes.append(
                f"実験の置き場 {value} の途中にリンクがあるので読みません"
            )
            continue
        if root.is_dir() and root not in roots:
            roots.append(root)

    return roots, notes


def child_dirs(folder: Path) -> list[Path]:
    """
    リンクでない子のディレクトリを返す関数

    Parameters
    ----------
    folder: Path
        ディレクトリ

    Returns
    ----------
    children: list[Path]
        名前の順の子のディレクトリ (読めなければ空)
    """
    # リンクはたどらない (置き場の外を読まないため)
    try:
        children = sorted(
            child for child in folder.iterdir()
            if child.is_dir() and not child.is_symlink()
        )
    except OSError:
        children = []

    return children


def read_yaml(path: Path) -> Any:
    """
    YAML のファイルを読む関数 (読めなければ None)

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    value: Any
        読んだ値 (無いか読めない場合は None)
    """
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        value = None

    return value


def read_copy(run_dir: Path) -> tuple[Optional[bytes], Optional[str]]:
    """
    実験の定義の写しを読み, 記録の sha256 と合うかを確かめる関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    data: Optional[bytes]
        写しの中身 (開けない場合は None)
    reason: Optional[str]
        開けない理由 (開ける場合は None)
    """
    # 学習を始めたときの定義かは, ファイルがあることでなく記録で見分ける
    # (ML-Found の docs/FORMAT.md)
    source = read_yaml(run_dir / "source.yaml")
    record = section(source, "model_spec")
    if record.get("status") != "copied" or not isinstance(
        record.get("sha256"), str
    ):
        return None, "定義の写しの記録がありません"
    path = run_dir / SPEC_COPY
    if path.is_symlink() or not path.is_file():
        return None, f"{SPEC_COPY} が普通のファイルではありません"
    try:
        if path.stat().st_size > MAX_SPEC_BYTES:
            return None, f"{SPEC_COPY} が 16 MiB を超えます"
        data = path.read_bytes()
    except OSError as error:
        return None, f"{SPEC_COPY} を読めません: {error}"
    if len(data) > MAX_SPEC_BYTES:
        return None, f"{SPEC_COPY} が 16 MiB を超えます"
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        return None, (
            f"{SPEC_COPY} の中身が記録 (学習を始めたときの定義) と違います"
        )

    return data, None


def suggested_name(spec_path: Optional[str], run: str) -> str:
    """
    保存する名前の候補を決める関数

    Parameters
    ----------
    spec_path: Optional[str]
        写した元の定義の場所
    run: str
        実験 (<名前>/<日付>/<時刻>)

    Returns
    ----------
    name: str
        元の定義の名前, 無ければ実験の名前 (どちらも規則に合わなければ
        model)
    """
    # 写しから起動し直した実験の元は model_spec.yaml なので実験の名前に
    stem = Path(spec_path).stem if spec_path else ""
    for candidate in (stem, run.split("/")[0]):
        if candidate != "model_spec" and NAME_PATTERN.fullmatch(candidate):
            return candidate
    name = "model"

    return name


def experiment_entry(root: Path, run_dir: Path) -> ExperimentEntry:
    """
    実験のディレクトリ 1 つを一覧の 1 件にする関数

    Parameters
    ----------
    root: Path
        実験の置き場
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    entry: ExperimentEntry
        実験と, 定義の写しを開けるか
    """
    run = run_dir.relative_to(root).as_posix()
    record = section(read_yaml(run_dir / "source.yaml"), "model_spec")
    spec_path = record.get("path")
    spec = spec_path if isinstance(spec_path, str) else None
    state = read_yaml(run_dir / "run.json")
    status = state.get("status") if isinstance(state, dict) else None
    _, reason = read_copy(run_dir)
    entry = ExperimentEntry(
        run=run, path=str(run_dir), spec=spec,
        name=suggested_name(spec, run),
        status=status if isinstance(status, str) else None,
        updated_at=modified_at(run_dir / SPEC_COPY)
        if (run_dir / SPEC_COPY).is_file() else modified_at(run_dir),
        usable=reason is None, reason=reason
    )

    return entry


def find_experiments(project: Path) -> ExperimentList:
    """
    プロジェクトの実験を探す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    found: ExperimentList
        実験の置き場と, 新しい順の実験
    """
    # 実験のディレクトリは <置き場>/<名前>/<日付>/<時刻>/ で, config と
    # source.yaml を持つもの (ML-Found の形)
    roots, notes = output_roots(project)
    entries: list[ExperimentEntry] = []
    for root in roots:
        for name_dir in child_dirs(root):
            for date_dir in child_dirs(name_dir):
                for run_dir in child_dirs(date_dir):
                    if all((run_dir / file).is_file() for file in RUN_FILES):
                        entries.append(experiment_entry(root, run_dir))
    entries.sort(key=lambda entry: entry.updated_at or "", reverse=True)
    found = ExperimentList(
        project=str(project), roots=[str(root) for root in roots],
        experiments=entries, notes=notes
    )

    return found


@router.post("/project/experiments")
def list_experiments(body: PathBody) -> ExperimentList:
    """
    プロジェクトの実験と, 定義の写しを開けるかの一覧を返す関数

    Parameters
    ----------
    body: PathBody
        プロジェクトのディレクトリか問題の設定のファイル

    Returns
    ----------
    found: ExperimentList
        実験の置き場と, 新しい順の実験
    """
    found = find_experiments(project_dir(body.path))

    return found


@router.post("/project/experiment-spec")
def open_experiment_spec(body: ExperimentBody) -> ExperimentSpec:
    """
    実験の定義の写しを読む関数

    Parameters
    ----------
    body: ExperimentBody
        プロジェクトの場所と実験

    Returns
    ----------
    opened: ExperimentSpec
        定義と保存する名前の候補

    Raises
    ----------
    HTTPException
        一覧に無い実験は 404, 写しを開けないか定義として読めない場合は 422
    """
    # 一覧に出す実験だけを読む (送られた文字から場所を組み立てない)
    found = find_experiments(project_dir(body.path))
    entry = next(
        (item for item in found.experiments if item.run == body.run), None
    )
    if entry is None:
        raise HTTPException(404, f"実験 {body.run} がありません")
    data, reason = read_copy(Path(entry.path))
    if data is None:
        raise HTTPException(422, f"実験 {body.run} の定義を開けません: {reason}")
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("定義が辞書ではありません")
        spec = load_spec(raw)
    except (ValueError, TypeError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise HTTPException(
            422, f"実験 {body.run} の定義を読めません: {error}"
        ) from error
    opened = ExperimentSpec(
        name=entry.name, spec=spec, run=entry.run, path=entry.path
    )

    return opened
