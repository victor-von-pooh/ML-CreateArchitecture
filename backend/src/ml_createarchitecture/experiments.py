"""
学習の実験に残ったモデルの定義の写しを開く API

ML-Found は実験のディレクトリ (<output_dir>/<名前>/<日付>/<時刻>/) に, 学習
に使った定義を model_spec.yaml として写し, 中身の sha256 を source.yaml の
model_spec に書く. 候補を比べて選んだ定義を, 名前を付けてプロジェクトの
models/ に保存し直すために, その写しを画面に開く. 読むのはプロジェクトの
中の実験の置き場だけで, リンクはたどらない
"""

import datetime as dt
import hashlib
import os
import stat
from collections.abc import Iterator
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

router = APIRouter()

# 実験の置き場を書く config (ML-Dashboard の get_output_dirs と同じ. 既定の
# config の値 (無ければ ML-Found の既定の outputs) に, 他の default の config
# と experiment の config の値を足す)
DEFAULT_DIR = Path("configs/default")
EXPERIMENT_DIR = Path("configs/experiment")
DEFAULT_OUTPUT = "outputs"

# 実験のディレクトリのファイル (ML-Found の docs/FORMAT.md)
SPEC_COPY = "model_spec.yaml"
RUN_FILES = ("config.yaml", "source.yaml")

# 読む大きさの上限 (写しは ML-Found が写す上限と同じ 16 MiB, config と記録
# は 1 MiB)
MAX_SPEC_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024

# 写しの入れ子の深さの上限 (ML-Dashboard が写しを使う決まりと同じ)
MAX_SPEC_DEPTH = 100
OPEN_EVENTS = (yaml.MappingStartEvent, yaml.SequenceStartEvent)
CLOSE_EVENTS = (yaml.MappingEndEvent, yaml.SequenceEndEvent)

# 一覧で辿るディレクトリの数の上限 (大きな置き場で長く止まらないため)
MAX_VISITS = 5000

# 実験の 3 段 (<名前>/<日付>/<時刻>)
RUN_DEPTH = 3

# リンクを辿らず (O_NOFOLLOW), 待たずに (O_NONBLOCK) 開く旗
FILE_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
)

# ML-Found の source.yaml の model_spec.status ごとの, 写していない理由
COPY_REASONS = {
    "missing": "学習の時に config の model.spec の定義が無かったので, "
    "写していません",
    "outside": "config の model.spec が学習のプロジェクトの外を指して"
    "いたので, "
    "写していません",
    "link": "config の model.spec の場所か途中がシンボリックリンクだった"
    "ので, 写していません",
    "not_file": "config の model.spec が普通のファイルでなかったので, "
    "写していません",
    "unreadable": "学習の時に config の model.spec の定義を読めなかった"
    "ので, 写していません",
    "too_large": "config の model.spec の定義が 16 MiB を超えていたので, "
    "写していません",
    "invalid": "config の model.spec が場所として読めない書き方だったので, "
    "写していません",
    "write_failed": "実験のディレクトリに書けなかったので, 写していません",
}


class ExperimentBody(BaseModel):
    path: str = Field(
        description="プロジェクトのディレクトリか問題の設定のファイルの絶対"
        "のパス (~ から書ける)"
    )
    run: str = Field(
        description="実験の置き場からの実験のディレクトリ (<名前>/<日付>/"
        "<時刻>. 一覧の run)"
    )
    root: Optional[str] = Field(
        default=None,
        description="実験の置き場 (一覧の root. 置き場が複数で run が重なる"
        "場合に見分ける)"
    )


class ExperimentEntry(BaseModel):
    # 実験と, その定義の写しを開けるか
    run: str = Field(description="実験の置き場からの <名前>/<日付>/<時刻>")
    root: str = Field(description="実験の置き場")
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
    # プロジェクト, 探した実験の置き場, 新しい順の実験, 読まなかった理由
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
            "ディレクトリ) ではありません. 学習のプロジェクトのディレクトリ"
            "か問題のファイルを選んでください"
        )
    found = from_problem(problem)
    if found is None or found.project is None:
        raise HTTPException(
            422, f"{problem.name} を選べません: {missing_record(problem)}"
        )
    project = Path(found.project)

    return project




def read_plain(path: Path, limit: int) -> Optional[bytes]:
    """
    リンクでない普通のファイルを, 上限の大きさまで読む関数

    確かめてから読むまでの間にリンクや FIFO に置き換えられても外を読まず
    止まらないように, リンクを辿らず待たずに開き, lstat で確かめたファイル
    と開いたファイルが同じ普通のファイルかを確かめてから, 上限 + 1 バイト
    まで読む (ML-Dashboard の評価のファイルの読み方)

    Parameters
    ----------
    path: Path
        ファイル
    limit: int
        読む大きさの上限 (バイト)

    Returns
    ----------
    content: Optional[bytes]
        中身 (無い, リンク, 普通でない, 上限を超える, 読めない場合は None)
    """
    try:
        found = path.lstat()
        if not stat.S_ISREG(found.st_mode) or found.st_size > limit:
            return None
        fd = os.open(path, FILE_FLAGS)
    except (OSError, ValueError):
        return None
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (
            (opened.st_dev, opened.st_ino) != (found.st_dev, found.st_ino)
        ):
            return None
        while total <= limit:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)
    content = b"".join(chunks) if total <= limit else None

    return content


def read_record(path: Path) -> Any:
    """
    小さな YAML (JSON も) のファイルを読む関数

    Parameters
    ----------
    path: Path
        config, source.yaml か run.json

    Returns
    ----------
    value: Any
        読んだ値 (無い, リンク, 1 MiB を超える, 読めない場合は None)
    """
    content = read_plain(path, MAX_RECORD_BYTES)
    if content is None:
        return None
    try:
        value = yaml.safe_load(content)
    except (yaml.YAMLError, RecursionError):
        value = None

    return value


def output_values(project: Path) -> list[str]:
    """
    config の experiment.output_dir の値を集める関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    values: list[str]
        既定の config の値 (無ければ outputs), 他の default の config と
        experiment の config の値
    """
    def value_of(path: Path) -> Optional[str]:
        value = section(read_record(path), "experiment").get("output_dir")
        text = value.strip() if isinstance(value, str) else ""
        return text or None

    # 別の config で出力先を変えた実験も探す (ML-Dashboard と同じ)
    values = [value_of(project / PROJECT_CONFIG) or DEFAULT_OUTPUT]
    files = sorted((project / DEFAULT_DIR).glob("*.yaml"))
    if (project / EXPERIMENT_DIR).is_dir():
        files += sorted((project / EXPERIMENT_DIR).rglob("*.yaml"))
    for file in files:
        value = value_of(file)
        if value and value not in values:
            values.append(value)

    return values


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
    # プロジェクトの中で, 途中にリンクの無い置き場だけを読む (決まった
    # 場所の外を読まないため)
    roots: list[Path] = []
    notes: list[str] = []
    for value in output_values(project):
        if "${" in value or "://" in value:
            notes.append(f"実験の置き場 {value} は参照か URL なので読みません")
            continue
        try:
            raw = Path(os.path.normpath(project / Path(value).expanduser()))
            root = raw.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if not root.is_relative_to(project) or root == project:
            notes.append(
                f"実験の置き場 {value} が学習のプロジェクトの外か, "
                "学習のプロジェクトそのものなので読みません"
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
            if not child.is_symlink() and child.is_dir()
        )
    except OSError:
        children = []

    return children


def is_run_dir(path: Path) -> bool:
    """
    ML-Found の実験のディレクトリ (config.yaml と source.yaml を持つ) か

    Parameters
    ----------
    path: Path
        ディレクトリ

    Returns
    ----------
    found: bool
        実験のディレクトリなら真
    """
    found = all(
        (path / file).is_file() and not (path / file).is_symlink()
        for file in RUN_FILES
    )

    return found


def copy_reason(record: dict[str, Any]) -> Optional[str]:
    """
    source.yaml の model_spec から, 写しを開けない理由を返す関数

    Parameters
    ----------
    record: dict[str, Any]
        source.yaml の model_spec (無ければ空)

    Returns
    ----------
    reason: Optional[str]
        写した記録が無い理由 (写した場合は None)
    """
    # 欄の無い古い記録と, 知らない値も写していないものとして読む
    status = record.get("status")
    if not record or status is None:
        return "定義の写しの記録がありません"
    if status == "copied" and isinstance(record.get("sha256"), str):
        return None
    reason = COPY_REASONS.get(str(status), "定義を写していません")

    return reason


def spec_record(run_dir: Path) -> dict[str, Any]:
    """
    実験の source.yaml の model_spec を読む関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    record: dict[str, Any]
        source.yaml の model_spec (無い, 読めない場合は空)
    """
    record = section(read_record(run_dir / "source.yaml"), "model_spec")

    return record


def read_copy(
    run_dir: Path, record: dict[str, Any]
) -> tuple[Optional[bytes], Optional[str]]:
    """
    実験の定義の写しを読み, 記録の sha256 と合うかを確かめる関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    record: dict[str, Any]
        source.yaml の model_spec (spec_record で読んだもの)

    Returns
    ----------
    data: Optional[bytes]
        写しの中身 (開けない場合は None)
    reason: Optional[str]
        開けない理由 (開ける場合は None)
    """
    # 学習を始めたときの定義かは, ファイルがあることでなく記録で見分ける
    # (ML-Found の docs/FORMAT.md)
    reason = copy_reason(record)
    if reason is not None:
        return None, reason
    path = run_dir / SPEC_COPY
    try:
        found = path.lstat()
    except FileNotFoundError:
        return None, f"{SPEC_COPY} がありません"
    except OSError:
        return None, f"{SPEC_COPY} を読めません"
    if stat.S_ISLNK(found.st_mode):
        return None, f"{SPEC_COPY} がシンボリックリンクです"
    if not stat.S_ISREG(found.st_mode):
        return None, f"{SPEC_COPY} が普通のファイルではありません"
    if found.st_size > MAX_SPEC_BYTES:
        return None, f"{SPEC_COPY} が 16 MiB を超えています"
    data = read_plain(path, MAX_SPEC_BYTES)
    if data is None:
        return None, f"{SPEC_COPY} を読めません"
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        return None, (
            f"{SPEC_COPY} の中身が記録 (学習を始めたときの定義) と違います"
        )

    return data, None


def parse_copy(data: bytes) -> dict[str, Any]:
    """
    写しの中身を定義として読む関数

    Parameters
    ----------
    data: bytes
        写しの中身

    Returns
    ----------
    spec: dict[str, Any]
        定義 (版を今の版に直したもの)

    Raises
    ----------
    ValueError
        YAML の別名を使う, 入れ子が深すぎる, YAML として読めない, 定義と
        して読めない場合
    """
    # 別名 (&a と *a) と深い入れ子は読む前の字句の段で断つ (自分を指す別名,
    # 別名を重ねて膨らむ定義, 深い入れ子で長く止まらないため. ML-Dashboard
    # が写しを使う決まりと同じ)
    depth = 0
    try:
        for event in yaml.parse(data, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent):
                raise ValueError(
                    "YAML の別名 (*) を使うので, 確かめずに開きません"
                )
            if isinstance(event, OPEN_EVENTS):
                depth += 1
                if depth > MAX_SPEC_DEPTH:
                    raise ValueError(
                        f"入れ子が {MAX_SPEC_DEPTH} 段を超えるので, YAML "
                        "として読めません"
                    )
            elif isinstance(event, CLOSE_EVENTS):
                depth -= 1
        raw = yaml.safe_load(data)
    except (yaml.YAMLError, RecursionError) as error:
        raise ValueError("YAML として読めません") from error
    if not isinstance(raw, dict):
        raise ValueError("定義が辞書ではありません")
    try:
        spec: dict[str, Any] = load_spec(raw)
    except (TypeError, KeyError, RecursionError) as error:
        raise ValueError(str(error)) from error

    return spec


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


def updated_time(run_dir: Path) -> Optional[str]:
    """
    一覧に出す実験の更新時刻を求める関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ

    Returns
    ----------
    stamp: Optional[str]
        定義の写しがリンクでない普通のファイルならその更新時刻, そうで
        なければ実験のディレクトリの更新時刻
    """
    # 写しは lstat で見て, リンクはたどらない (他の読みと同じ決まり)
    try:
        found = (run_dir / SPEC_COPY).lstat()
    except OSError:
        found = None
    if found is None or not stat.S_ISREG(found.st_mode):
        stamp = modified_at(run_dir)
        return stamp
    moment = dt.datetime.fromtimestamp(found.st_mtime)
    stamp = moment.astimezone().isoformat(timespec="seconds")

    return stamp


def experiment_entry(
    root: Path, run_dir: Path, record: dict[str, Any],
    reason: Optional[str]
) -> ExperimentEntry:
    """
    実験のディレクトリ 1 つを一覧の 1 件にする関数

    Parameters
    ----------
    root: Path
        実験の置き場
    run_dir: Path
        実験のディレクトリ
    record: dict[str, Any]
        source.yaml の model_spec (spec_record で読んだもの)
    reason: Optional[str]
        写しを開けない理由 (開ける場合は None)

    Returns
    ----------
    entry: ExperimentEntry
        実験と, 定義の写しを開けるか
    """
    run = run_dir.relative_to(root).as_posix()
    spec_path = record.get("path")
    spec = spec_path if isinstance(spec_path, str) else None
    state = read_record(run_dir / "run.json")
    status = state.get("status") if isinstance(state, dict) else None
    entry = ExperimentEntry(
        run=run, root=str(root), path=str(run_dir), spec=spec,
        name=suggested_name(spec, run),
        status=status if isinstance(status, str) else None,
        updated_at=updated_time(run_dir),
        usable=reason is None, reason=reason
    )

    return entry


def walk_runs(roots: list[Path], notes: list[str]) -> Iterator[
    tuple[Path, Path]
]:
    """
    実験の置き場の 3 段を辿り, 実験のディレクトリを返す関数

    Parameters
    ----------
    roots: list[Path]
        実験の置き場
    notes: list[str]
        辿る数の上限で打ち切った知らせを足すリスト

    Returns
    ----------
    runs: Iterator[tuple[Path, Path]]
        置き場と実験のディレクトリ
    """
    # 辿るディレクトリの数に上限 (大きな置き場で長く止まらないため)
    visits = 0
    for root in roots:
        level = [root]
        for _ in range(RUN_DEPTH):
            next_level: list[Path] = []
            for folder in level:
                for child in child_dirs(folder):
                    visits += 1
                    if visits > MAX_VISITS:
                        notes.append(
                            f"ディレクトリが {MAX_VISITS} を超えるので, "
                            "途中までの実験だけを並べます"
                        )
                        return
                    next_level.append(child)
            level = next_level
        for run_dir in level:
            if is_run_dir(run_dir):
                yield root, run_dir


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
    # source.yaml は実験 1 件ごとに 1 回だけ読む
    entries: list[ExperimentEntry] = []
    for root, run_dir in walk_runs(roots, notes):
        record = spec_record(run_dir)
        entries.append(
            experiment_entry(
                root, run_dir, record, read_copy(run_dir, record)[1]
            )
        )
    entries.sort(key=lambda entry: entry.updated_at or "", reverse=True)
    found = ExperimentList(
        project=str(project), roots=[str(root) for root in roots],
        experiments=entries, notes=notes
    )

    return found


def find_run(
    project: Path, run: str, root_text: Optional[str]
) -> tuple[Path, Path]:
    """
    置き場の 3 段を名前で辿り, 一覧の run の実験のディレクトリを探す関数

    送られた文字は, 置き場の中の子のディレクトリの名前と比べるだけで, 場所
    を組み立てない

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    run: str
        <名前>/<日付>/<時刻>
    root_text: Optional[str]
        実験の置き場 (無ければ全ての置き場から探す)

    Returns
    ----------
    found: tuple[Path, Path]
        置き場と実験のディレクトリ

    Raises
    ----------
    HTTPException
        無い場合は 404, 置き場が複数で run が重なり置き場を送っていない
        場合は 409
    """
    parts = run.split("/")
    roots, _ = output_roots(project)
    if root_text is not None:
        roots = [root for root in roots if str(root) == root_text]
    matches: list[tuple[Path, Path]] = []
    if len(parts) == RUN_DEPTH:
        for root in roots:
            folder: Optional[Path] = root
            for part in parts:
                assert folder is not None
                folder = next(
                    (child for child in child_dirs(folder)
                     if child.name == part), None
                )
                if folder is None:
                    break
            if folder is not None and is_run_dir(folder):
                matches.append((root, folder))
    if not matches:
        raise HTTPException(404, f"実験 {run} がありません")
    if len(matches) > 1:
        raise HTTPException(
            409, f"実験 {run} が複数の置き場にあります. 置き場を選んでください"
        )
    found = matches[0]

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
        プロジェクトの場所, 実験と置き場

    Returns
    ----------
    opened: ExperimentSpec
        定義と保存する名前の候補

    Raises
    ----------
    HTTPException
        実験が無い場合は 404, 置き場が重なる場合は 409, 写しを開けないか
        定義として読めない場合は 422
    """
    root, run_dir = find_run(project_dir(body.path), body.run, body.root)
    record = spec_record(run_dir)
    data, reason = read_copy(run_dir, record)
    if data is None:
        raise HTTPException(422, f"実験 {body.run} の定義を開けません: {reason}")
    try:
        spec = parse_copy(data)
    except ValueError as error:
        raise HTTPException(
            422, f"実験 {body.run} の定義を読めません: {error}"
        ) from error
    entry = experiment_entry(root, run_dir, record, None)
    opened = ExperimentSpec(
        name=entry.name, spec=spec, run=entry.run, path=entry.path
    )

    return opened
