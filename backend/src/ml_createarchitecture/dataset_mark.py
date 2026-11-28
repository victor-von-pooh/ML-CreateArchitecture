"""
モデルの定義の隣に残す, データセットの印

定義を保存するときに, 元にしたデータセットの dataset.yaml の sha256 を
models/<名前>.dataset.json に残す. 定義の中には入れない (ML-ModelControl が
知らない項目を誤りにし, 定義の sha256 も替わるため). 学習・評価の段階と
進みが, データセットを書き出し直した後に定義が古いかを比べるために読む
"""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Optional

from ml_createarchitecture.store import FILE_MODE

# 印のファイルの拡張子 (YAML でない. 学習・評価の段階が models/ の下の YAML
# を定義の候補に並べるため)
MARK_SUFFIX = ".dataset.json"

# データセットのディレクトリが持つ説明のファイルの名前
DATASET_FILE = "dataset.yaml"

# 一定の大きさずつ読む大きさ
CHUNK_BYTES = 1 << 20

# 印のファイルとして読む大きさの上限
MAX_MARK_BYTES = 1 << 16

# 末尾が区切りの書き方 (table.csv/) はディレクトリの書き方
TRAILING_SEPARATORS = ("/", "/.", os.sep, os.sep + ".")

# 印の形の版
MARK_VERSION = 1


def open_regular(path: Path) -> tuple[Optional[int], Optional[str]]:
    """
    普通のファイルを待たずに開く関数

    FIFO で止まらないよう O_NONBLOCK で開き, 開いたものが普通のファイルか
    を開いた後に確かめる (確かめた後の入れ替えに備えるため)

    Parameters
    ----------
    path: Path
        ファイルの場所 (リンクは辿る)

    Returns
    ----------
    fd: Optional[int]
        開いた記述子 (開けなかった場合は None)
    reason: Optional[str]
        開けなかった理由 (missing, invalid, not_file, unreadable)
    """
    # 開けない理由の分類
    flags = (
        os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError):
        return None, "missing"
    except ValueError:
        return None, "invalid"
    except OSError:
        return None, "unreadable"

    # 普通のファイルでなければ閉じる処理
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
    except OSError:
        os.close(fd)
        return None, "unreadable"
    if not regular:
        os.close(fd)
        return None, "not_file"

    return fd, None


def hash_file(path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    普通のファイルの中身の sha256 を, 全体を持たずに求める関数

    Parameters
    ----------
    path: Path
        ファイルの場所 (リンクは辿る)

    Returns
    ----------
    sha256: Optional[str]
        中身の SHA-256 (16 進の小文字), 求められなかった場合は None
    reason: Optional[str]
        求められなかった理由 (missing, invalid, not_file, unreadable),
        求めた場合は None
    """
    # 一定の大きさずつ読み, 途中の誤りは読めないものと判定
    fd, reason = open_regular(path)
    if fd is None:
        return None, reason
    digest = hashlib.sha256()
    try:
        while chunk := os.read(fd, CHUNK_BYTES):
            digest.update(chunk)
    except (OSError, ValueError):
        return None, "unreadable"
    finally:
        os.close(fd)

    return digest.hexdigest(), None


def climb_reason(path: Path) -> Optional[str]:
    """
    正規化する前の場所の .. が, OS の辿り方と同じ場所に上がるかを確かめる
    関数

    Parameters
    ----------
    path: Path
        正規化する前の絶対パス

    Returns
    ----------
    reason: Optional[str]
        .. の直前がリンクなら link (正規化で .. を消すと OS が開く場所と
        違う場所を求めるため), 無いかディレクトリでなければ missing. 問題が
        無ければ None
    """
    # 先頭から 1 つずつ辿り, .. の直前の場所を確認
    current = Path(path.anchor)
    for part in path.parts[1:]:
        if part != "..":
            current = current / part
            continue
        if current.is_symlink():
            return "link"
        if not current.is_dir():
            return "missing"
        current = current.parent

    return None


def dataset_sha256(
    value: str, project: Path
) -> tuple[Optional[str], Optional[str]]:
    """
    データセットの場所から, 印にする sha256 を求める関数

    ディレクトリは中の dataset.yaml, ファイル 1 つのデータセットはその
    ファイルの中身の sha256 にする

    Parameters
    ----------
    value: str
        データセットの場所 (相対ならプロジェクトから. ~ から書ける)
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    sha256: Optional[str]
        中身の SHA-256, 求められなかった場合は None
    reason: Optional[str]
        求められなかった理由 (missing, invalid, not_file, unreadable)
    """
    # リンクの後の .. は正規化で別の場所になるので求めない
    try:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raw = project / raw
        reason = climb_reason(raw)
        path = Path(os.path.normpath(raw))
        is_dir = path.is_dir()
    except ValueError:
        return None, "invalid"
    except (OSError, RuntimeError):
        return None, "unreadable"
    if reason == "link":
        return None, "invalid"
    if reason is not None:
        return None, reason

    # 末尾が区切りの書き方はディレクトリの書き方なので, ファイルがあっても
    # 普通のファイルとは見なさない
    if not is_dir and value.endswith(TRAILING_SEPARATORS):
        return None, "not_file" if path.exists() else "missing"
    target = path / DATASET_FILE if is_dir else path

    return hash_file(target)


def mark_path(spec_path: Path) -> Path:
    """
    モデルの定義の隣の印のファイルの場所を返す関数

    Parameters
    ----------
    spec_path: Path
        モデルの定義のファイル (models/<名前>.yaml)

    Returns
    ----------
    path: Path
        models/<名前>.dataset.json
    """
    path = spec_path.with_name(spec_path.stem + MARK_SUFFIX)

    return path


def remove_mark(spec_path: Path) -> None:
    """
    定義の隣の印を消す関数 (無いか消せなければ何もしない)

    Parameters
    ----------
    spec_path: Path
        モデルの定義のファイル

    Returns
    ----------
    None
    """
    # 消せない場合は印の中の定義の sha256 が合わず無視される
    try:
        mark_path(spec_path).unlink(missing_ok=True)
    except OSError:
        pass


def write_mark(
    spec_path: Path, dataset: Optional[str], project: Path
) -> Optional[str]:
    """
    書いた定義の隣に, データセットの印を残す関数

    データセットを読めない場合は印を残さず前の印を消し (読めない知らせは
    骨組みの約束の確かめが出す), 印を書けない場合は前の印を消して理由を
    返す. どちらも例外は出さない (保存を落とさない)

    Parameters
    ----------
    spec_path: Path
        書いたモデルの定義のファイル
    dataset: Optional[str]
        記録のデータセットの場所 (無ければ印を残さない)
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    note: Optional[str]
        印を書けなかった理由の文 (残した場合と, 残さない場合は None)
    """
    # 記録が無い場合は前の印の削除だけ
    if not dataset:
        remove_mark(spec_path)
        return None

    # 書いた定義の sha256 は, 書いたファイルを読んで求める (学習・評価の
    # 段階が求める値と同じにするため)
    sha256, _ = dataset_sha256(dataset, project)
    spec_sha256, _ = hash_file(spec_path)
    if sha256 is None or spec_sha256 is None:
        remove_mark(spec_path)
        return None
    record = {
        "format_version": MARK_VERSION,
        "dataset": {"path": dataset, "sha256": sha256},
        "spec_sha256": spec_sha256,
    }
    target = mark_path(spec_path)
    temp: Optional[str] = None
    try:
        # 隣がリンクなら書かない (リンクの先を書き換えないため)
        if target.is_symlink():
            raise OSError("リンクです")
        handle, temp = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp"
        )
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, indent=2))
            file.write("\n")
        os.chmod(temp, FILE_MODE)
        Path(temp).replace(target)
    except OSError as error:
        remove_mark(spec_path)
        return f"データセットの印を書けません: {error}"
    finally:
        # 置き換えた後も失敗した後も一時ファイルは残さない
        if temp is not None:
            Path(temp).unlink(missing_ok=True)

    return None


def read_mark(spec_path: Path) -> Optional[dict[str, Any]]:
    """
    定義の隣の印を, 今の定義の印として読む関数

    印が無い, 読めない, 形が違う, 印と一緒に書いた定義の sha256 が今の定義
    と合わない (定義を他で書き換えた) 場合は, 印が無い物として None

    Parameters
    ----------
    spec_path: Path
        モデルの定義のファイル

    Returns
    ----------
    dataset: Optional[dict[str, Any]]
        印のデータセット (path, sha256). 印が無い物の場合は None
    """
    # 小さいファイルだけを待たずに読む
    fd, _ = open_regular(mark_path(spec_path))
    if fd is None:
        return None
    try:
        data = os.read(fd, MAX_MARK_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > MAX_MARK_BYTES:
        return None
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None

    # 形と今の定義の印かの確認
    if not isinstance(record, dict):
        return None
    dataset = record.get("dataset")
    if (
        type(record.get("format_version")) is not int
        or record["format_version"] != MARK_VERSION
        or not isinstance(dataset, dict)
        or not isinstance(dataset.get("path"), str)
        or not isinstance(dataset.get("sha256"), str)
        or not isinstance(record.get("spec_sha256"), str)
    ):
        return None
    current, _ = hash_file(spec_path)
    if current is None or current != record["spec_sha256"]:
        return None

    return {"path": dataset["path"], "sha256": dataset["sha256"]}
