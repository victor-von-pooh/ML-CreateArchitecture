"""
データセットとプロジェクトにつなぐ API

データセットの説明は ML-EDA, 問題の設定は ML-Quest を通して読む (他の段階
のファイルは, その約束を持つライブラリを通して読む決まり). どちらも追加の
依存 project で入る
"""

import datetime as dt
import functools
import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException
from ml_modelcontrol import list_examples
from pydantic import BaseModel, Field, ValidationError

from ml_createarchitecture.settings import get_problems_dir
from ml_createarchitecture.store import (
    NAME_PATTERN,
    ModelError,
    spec_text,
    write_atomic,
)

router = APIRouter(prefix="/api")

# 追加の依存の入れ方 (無い場合の案内)
SETUP_HINT = "make setup (EXTRAS を書く場合は project も含める)"

# 形の推定の例に使う, 決まらない次元 (バッチなど) の大きさ (1 はブロード
# キャストで形の誤りを見逃すので使わない)
EXAMPLE_SIZE = 4

# 問題の設定の記録が無いプロジェクトを見分ける config と, その既定の保存先
# (ML-Quest の骨組みの config の model.spec)
PROJECT_CONFIG = Path("configs/default/train.yaml")
DEFAULT_SPEC = Path("models/model.yaml")

# 名前で選ぶ保存先のディレクトリ (ML-Dashboard のモデルの定義の候補の場所)
MODELS_DIR = "models"

# プロジェクトのディレクトリの問題の設定 (ML-Pathfinder の置き場でも同じ
# 名前)
PROBLEM_FILE = "problem.yaml"

# 学習の設定を置くプロジェクトのディレクトリ (モデルの定義は書かない)
CONFIG_DIR = "configs"

# モデルの定義を書かない, プロジェクトの直下のディレクトリと理由 (ML-Quest
# が写したデータ, ML-EDA のデータセットと ML-Found の実験の置き場)
RESERVED_DIRS = {
    "data": "写したデータのディレクトリ",
    "outputs": "データセットと実験のディレクトリ",
}

# 書けるモデルの定義の拡張子
SPEC_SUFFIXES = (".yaml", ".yml")

# 他の段階のファイルを読めないときの例外 (ライブラリの中の形の崩れも
# 含む)
READ_ERRORS = (
    OSError, ValueError, TypeError, KeyError, AttributeError,
    yaml.YAMLError
)


class PathBody(BaseModel):
    path: str = Field(description="絶対のパス (~ から書ける)")


class DatasetBody(BaseModel):
    path: str = Field(description="データセットの絶対のパス (~ から書ける)")
    problem: Optional[str] = Field(
        default=None,
        description="問題の設定のファイルの絶対のパス (表の生成なら, 入力と"
        "出力の幅を ML-Quest の table_encoding の符号化した行の幅にする. "
        "並べるなら, path は反応の表のファイルで, 入力 user と item の大きさ"
        "を ML-Quest の rank_encoding の学習用の利用者と品物の数にする. "
        "異常の検知なら, 度合いの決め方を受け持ちの答えから読む)"
    )


class TargetBody(BaseModel):
    path: str = Field(
        description="プロジェクトのディレクトリか問題の設定のファイルの絶対"
        "のパス (~ から書ける)"
    )
    name: Optional[str] = Field(
        default=None,
        description="models/ の下の名前 (拡張子無し. 無いか空なら既定)"
    )


class DatasetInput(BaseModel):
    # モデルの入力の名前, 形の例 (バッチの次元を含む), 型と, データセットの
    # 形 (決まらない次元は null)
    name: str
    shape: list[int]
    dtype: str
    dataset_shape: list[Optional[int]] = Field(
        default_factory=list,
        description="model_inputs の形 (決まらない次元は null)"
    )
    vocabulary: Optional[int] = Field(
        default=None,
        description="文の語の番号の入力の語彙の大きさ (埋め込みの数)"
    )


class TableWidth(BaseModel):
    # 表の生成の符号化した行の幅と, その内訳 (ML-Quest の table_encoding)
    width: int = Field(description="符号化した行の幅")
    numbers: int = Field(description="数の列の数 (1 ずつ)")
    categories: int = Field(description="種類の列の数")
    values: int = Field(description="種類の列の値の種類の数の和")
    columns: int = Field(description="学習に使う列の数")
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )


class RankSize(BaseModel):
    # 並べるの学習用の利用者と品物の数と, その決め方 (ML-Quest の
    # rank_encoding)
    users: int = Field(description="学習用の利用者の数 (入力 user の大きさ)")
    items: int = Field(description="学習用の品物の数 (入力 item の大きさ)")
    rows: dict[str, int] = Field(
        description="分割の名前から反応の行の数 (時刻の順に分けたもの)"
    )
    valid: Optional[float] = Field(
        default=None,
        description="プロジェクトの config の検証用の反応の行の割合 (書いて"
        "いなければ無し. 学習のスクリプトの既定)"
    )
    test: Optional[float] = Field(
        default=None,
        description="プロジェクトの config のテスト用の反応の行の割合 (書いて"
        "いなければ無し. 学習のスクリプトの既定)"
    )
    notes: list[str] = Field(
        default_factory=list, description="除いた行の知らせ"
    )
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )


class AnomalyWidth(BaseModel):
    # 異常の検知の入力と出力の幅 (学習に使う列の数) と, その決め方
    columns: Optional[int] = Field(
        description="学習に使う列の数 (自己符号化器の入力と出力の幅. 正解の"
        "列と行の id の列は数えない)"
    )
    target: Optional[str] = Field(
        default=None,
        description="異常の正解の列 (正解の無い表は無し)"
    )
    score: Optional[str] = Field(
        default=None,
        description="問題の設定の度合いの決め方 (autoencoder か distance. "
        "問題の設定を渡していなければ無し)"
    )
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )


class DatasetInputs(BaseModel):
    # データセットの場所と, ML-EDA の model_inputs の結果
    path: str
    inputs: list[DatasetInput]
    task: Optional[str] = None
    target: Optional[str] = None
    classes: Optional[list[Any]] = None
    target_kind: Optional[str] = Field(
        default=None,
        description="ファイルの組の正解の種類 (mask, point_cloud など)"
    )
    outputs: Optional[int] = None
    table: Optional[TableWidth] = Field(
        default=None,
        description="表の生成の問題の設定で, 入力と出力の幅にした符号化した"
        "行の幅 (他は無し)"
    )
    rank: Optional[RankSize] = Field(
        default=None,
        description="並べるの問題の設定で, 入力 user と item の大きさにした"
        "学習用の利用者と品物の数 (他は無し)"
    )
    anomaly: Optional[AnomalyWidth] = Field(
        default=None,
        description="異常の検知のデータセットか問題の設定で, 入力と出力の幅"
        "にした学習に使う列の数と度合いの決め方 (他は無し)"
    )


class ProjectTarget(BaseModel):
    # モデルの定義の保存先と, どこから決めたか
    path: str
    source: str = Field(
        description="problem (問題の設定の記録), project (config の "
        "model.spec) か name (models/ の下の名前)"
    )
    exists: bool
    project: Optional[str] = None
    dataset: Optional[str] = None
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )
    problem_path: Optional[str] = Field(
        default=None,
        description="記録を読んだ問題の設定のファイル (データセットから入力と"
        "出力を作るときに渡す)"
    )
    example: Optional[str] = Field(
        default=None,
        description="問題の設定の受け持ちに合う雛形 (ML-ModelControl の"
        "同梱の例の名前. 合うものが無ければ無し)"
    )
    default: bool = Field(
        default=False,
        description="名前の保存先が既定の保存先 (記録か config の "
        "model.spec) と同じファイルか"
    )


class ProblemEntry(BaseModel):
    # ML-Pathfinder の置き場の問題の設定の 1 件と, そのプロジェクトの記録
    path: str = Field(description="problem.yaml の場所")
    name: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    usable: bool = Field(description="保存先を決められるか")
    reason: Optional[str] = Field(
        default=None, description="使えない理由"
    )
    project: Optional[str] = None
    dataset: Optional[str] = None
    target: Optional[str] = Field(
        default=None, description="モデルの定義の保存先"
    )
    example: Optional[str] = Field(
        default=None,
        description="受け持ちに合う雛形 (ML-ModelControl の同梱の例の名前)"
    )


class ProblemList(BaseModel):
    # 置き場の problems, 問題の設定 (更新の新しい順), 置き場を読めない理由
    root: str
    problems: list[ProblemEntry]
    error: Optional[str] = None


class ProjectSaveBody(BaseModel):
    # 保存先を決める場所, models/ の下の名前, 定義, 既にある場合に上書き
    # するか
    path: str
    name: Optional[str] = Field(
        default=None,
        description="models/ の下の名前 (拡張子無し. 無いか空なら既定)"
    )
    spec: dict[str, Any]
    overwrite: bool = False
    target: Optional[str] = Field(
        default=None,
        description="確かめた保存先 (POST /api/project/target の path). 送る"
        "と, 書くときに決め直した保存先と違えば 409"
    )


class ProjectSaved(BaseModel):
    # 書いたファイルと更新時刻
    path: str
    updated_at: str


def load_library(name: str, attribute: str) -> ModuleType:
    """
    追加の依存のライブラリを読み込む関数

    Parameters
    ----------
    name: str
        モジュールの名前 (ml_eda か ml_quest)
    attribute: str
        使う関数の名前 (古い版で無い場合を見分けるため)

    Returns
    ----------
    module: ModuleType
        読み込んだモジュール

    Raises
    ----------
    HTTPException
        入っていない場合と, 使う関数の無い古い版の場合は 501
    """
    # 入っていなければ入れ方, 古ければ入れ直し方の案内
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise HTTPException(
            501,
            f"{name} が入っていません. {SETUP_HINT} で入れてから起動し直して"
            "ください"
        ) from error
    if not hasattr(module, attribute):
        raise HTTPException(
            501,
            f"{name} が古く, {attribute} がありません. {SETUP_HINT} で入れ"
            "直してから起動し直してください"
        )

    return module


def absolute_path(text: str) -> Path:
    """
    画面から受け取った場所を絶対のパスにする関数

    Parameters
    ----------
    text: str
        絶対のパス (~ から書ける)

    Returns
    ----------
    path: Path
        リンクをたどった絶対のパス

    Raises
    ----------
    HTTPException
        空, 相対のパス, パスとして読めない場合は 422
    """
    # 起動した場所に依る相対のパスは受け取らない
    try:
        raw = Path(text.strip()).expanduser()
        if not text.strip() or not raw.is_absolute():
            raise HTTPException(
                422, "場所は絶対のパス (/ か ~ から) で書いてください"
            )
        path = raw.resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            422, f"場所として読めません: {error}"
        ) from error

    return path


def positive_int(value: Any) -> Optional[int]:
    """
    正の整数ならそのまま, 他は None を返す関数

    Parameters
    ----------
    value: Any
        model_inputs の値 (語彙の大きさなど)

    Returns
    ----------
    number: Optional[int]
        正の整数 (真偽は除く), 他は None
    """
    # 真偽は int の仲間なので除く
    number = (
        value if isinstance(value, int) and not isinstance(value, bool)
        and value > 0 else None
    )

    return number


def example_shape(shape: Any) -> list[int]:
    """
    データセットの入力の形を, 形の推定に使う例の形にする関数

    Parameters
    ----------
    shape: Any
        model_inputs の形 (決まらない次元は None)

    Returns
    ----------
    example: list[int]
        決まらない次元を例の大きさにした形

    Raises
    ----------
    HTTPException
        形が整数と None の並びでない場合は 422
    """
    # 決まらない次元 (バッチなど) は例の大きさ, 他は正の整数だけ
    if not isinstance(shape, list):
        raise HTTPException(422, f"入力の形 {shape!r} を読めません")
    example: list[int] = []
    for size in shape:
        if size is None:
            example.append(EXAMPLE_SIZE)
        elif isinstance(size, int) and not isinstance(size, bool) and (
            size > 0
        ):
            example.append(size)
        else:
            raise HTTPException(422, f"入力の形 {shape!r} を読めません")

    return example


def read_problem(problem_path: Path) -> Any:
    """
    データセットから入力と出力を作るときに渡された問題の設定を読む関数

    Parameters
    ----------
    problem_path: Path
        問題の設定のファイル

    Returns
    ----------
    problem: Any
        ML-Quest の load_problem で読んだ問題の設定

    Raises
    ----------
    HTTPException
        ML-Quest が無い場合は 501, ファイルが無い場合は 404, 読めない場合は
        422
    """
    # 読み方と版の確かめは ML-Quest に任せる
    if not problem_path.is_file():
        raise HTTPException(404, f"問題の設定 {problem_path} がありません")
    quest = load_library("ml_quest", "load_problem")
    try:
        problem = quest.load_problem(problem_path)
    except READ_ERRORS as error:
        raise HTTPException(
            422, f"問題の設定を読めません: {error}"
        ) from error

    return problem


def problem_name(problem: Any) -> Optional[str]:
    """
    問題の設定の名前を返す関数

    Parameters
    ----------
    problem: Any
        問題の設定

    Returns
    ----------
    name: Optional[str]
        名前 (文字でなければ None)
    """
    # 手で崩した値は使わない
    value = problem.get("name") if isinstance(problem, dict) else None
    name = value if isinstance(value, str) else None

    return name


def table_width(problem: Any, dataset: Path) -> Optional[TableWidth]:
    """
    表の生成の問題の設定なら, データセットの符号化した行の幅を求める関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)
    dataset: Path
        データセットのディレクトリ

    Returns
    ----------
    table: Optional[TableWidth]
        符号化した行の幅と内訳 (表の生成でなければ None)

    Raises
    ----------
    HTTPException
        ML-Quest が古い場合は 501, 幅を求められない場合 (目的の列か欠損の
        ある列がある, 種類の値が多すぎる列があるなど) は 422
    """
    # 列の符号化の決まりは ML-Quest に任せる (学習のスクリプトと同じ幅に
    # するため)
    quest = load_library("ml_quest", "table_encoding")
    try:
        encoding = quest.table_encoding(problem, dataset)
    except READ_ERRORS as error:
        raise HTTPException(
            422,
            "表の生成の, 学習で使う 1 行の長さ (入力と出力の幅) を求められ"
            f"ません: {error}"
        ) from error
    if encoding is None:
        table = None
        return table

    # 幅と内訳 (形の崩れた値は 422)
    try:
        columns = encoding["columns"]
        values = sum(
            len(column["categories"]) for column in columns
            if column["type"] == "category"
        )
        table = TableWidth(
            width=encoding["width"], numbers=encoding["numbers"],
            categories=encoding["categories"], values=values,
            columns=len(columns), problem=problem_name(problem)
        )
    except (*READ_ERRORS, ValidationError) as error:
        raise HTTPException(
            422,
            f"学習で使う 1 行の長さ (入力と出力の幅) を読めません: {error}"
        ) from error
    if table.width < 1:
        raise HTTPException(
            422,
            "表の生成の, 学習で使う 1 行の長さ (入力と出力の幅) が 0 です "
            "(学習に使う項目がありません)"
        )

    return table


def split_rates(problem: Any) -> tuple[dict[str, float], list[str]]:
    """
    問題の設定が記録したプロジェクトの config から, 反応を時刻の順に分ける
    割合を読む関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    rates: dict[str, float]
        valid と test のうち config に数で書いてあるもの (プロジェクトの
        記録か config が無いか読めなければ空. 学習のスクリプトの既定に
        任せる)
    notes: list[str]
        config があるのに読めない場合の, 既定の割合に戻した知らせ (他は
        空)
    """
    # 既定に任せるだけなので, 読めない config は止めずに知らせる
    quest = load_library("ml_quest", "project_of")
    try:
        place = quest.project_of(problem).get("path")
    except READ_ERRORS:
        place = None
    rates: dict[str, float] = {}
    notes: list[str] = []
    if not isinstance(place, str) or not (
        Path(place) / PROJECT_CONFIG
    ).is_file():
        return rates, notes
    try:
        config = yaml.safe_load(
            (Path(place) / PROJECT_CONFIG).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        config = None
    if not isinstance(config, dict):
        notes.append(
            f"学習の設定 ({PROJECT_CONFIG}) を読めないので, 分ける割合は"
            "既定の値です"
        )
        return rates, notes
    data = section(config, "data")
    for key in ("valid", "test"):
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rates[key] = float(value)

    return rates, notes


def is_rank(problem: Any) -> bool:
    """
    問題の設定の受け持ちが並べるかを返す関数

    並べる以外では rank_encoding を求めない (rank_encoding の無い古い
    ML-Quest でも, 表の生成や列の予測を止めないため)

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    rank: bool
        受け持ちの答えが並べる (rank) か (要らなくなった問いの古い答えは
        除外. 読めなければ False)
    """
    # 答えの読み方は ML-Quest に任せる
    quest = load_library("ml_quest", "active_value")
    try:
        role = quest.active_value(problem, f"{quest.PART}.role")
    except READ_ERRORS:
        role = None
    rank = bool(isinstance(role, str) and role == "rank")

    return rank


def anomaly_score(problem: Any) -> Optional[str]:
    """
    問題の設定の受け持ちが異常の検知なら, 度合いの決め方を返す関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    score: Optional[str]
        度合いの決め方 (autoencoder か distance. 答えていなければ空の文字.
        予測するで正解の種類が異常, 作り方が既製のモデルをそのままでない
        受け持ちでなければ None. 要らなくなった問いの古い答えは除外)
    """
    # 答えの読み方は ML-Quest に任せる (受け持ちの見分け方は ML-Quest の
    # 異常の検知の受け持ちと同じ)
    quest = load_library("ml_quest", "active_value")
    try:
        role = quest.active_value(problem, f"{quest.PART}.role")
        kind = quest.active_value(problem, f"{quest.PART}.target.kind")
        method = quest.active_value(problem, f"{quest.PART}.method")
        value = quest.active_value(problem, f"{quest.PART}.anomaly.score")
    except READ_ERRORS:
        role = kind = method = value = None
    if role != "predict" or kind != "anomaly" or method == "pretrained":
        score = None
        return score
    score = value if isinstance(value, str) else ""

    return score


def rank_size(problem: Any, dataset: Path) -> Optional[RankSize]:
    """
    並べるの問題の設定なら, 反応の表から学習用の利用者と品物の数を求める
    関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)
    dataset: Path
        反応の表のファイル (学習のスクリプトの config の data.dataset)

    Returns
    ----------
    rank: Optional[RankSize]
        学習用の利用者と品物の数と決め方 (並べるでなければ None. 並べるの
        受け持ちかは is_rank で先に見て, 並べる以外では ML-Quest の
        rank_encoding を読まない)

    Raises
    ----------
    HTTPException
        並べるで ML-Quest が古い場合は 501, 反応の表が無い場合は 404,
        数を求められない場合 (列が無い, 時刻を読めない, 学習用か検証用が
        空になるなど) は 422
    """
    # 読み方, 分割と番号の決まりは ML-Quest に任せる (学習のスクリプトと
    # 同じ数にするため). 割合はプロジェクトの config にあればそれ
    if not is_rank(problem):
        rank = None
        return rank
    quest = load_library("ml_quest", "rank_encoding")
    rates, notes = split_rates(problem)
    try:
        encoding = quest.rank_encoding(problem, dataset, **rates)
    except FileNotFoundError as error:
        raise HTTPException(
            404, f"反応の表がありません: {error}"
        ) from error
    except READ_ERRORS as error:
        raise HTTPException(
            422,
            "並べるの, 学習用の利用者と品物の数 (入力 user と item の大きさ)"
            f" を求められません: {error}"
        ) from error
    if encoding is None:
        rank = None
        return rank

    # 数と決め方 (形の崩れた値は 422)
    try:
        rank = RankSize(
            users=encoding["users"], items=encoding["items"],
            rows=encoding["rows"], valid=rates.get("valid"),
            test=rates.get("test"),
            notes=[*notes, *(encoding.get("notes") or [])],
            problem=problem_name(problem)
        )
    except (*READ_ERRORS, ValidationError) as error:
        raise HTTPException(
            422,
            f"学習用の利用者と品物の数を読めません: {error}"
        ) from error
    if rank.users < 1 or rank.items < 1:
        raise HTTPException(
            422, "学習用の反応に利用者か品物がいません"
        )

    return rank


def rank_inputs(path: Path, rank: RankSize) -> DatasetInputs:
    """
    並べるの 2 塔のモデルの入力 (利用者と品物の番号のワンホット) を作る
    関数

    Parameters
    ----------
    path: Path
        反応の表のファイル
    rank: RankSize
        学習用の利用者と品物の数

    Returns
    ----------
    result: DatasetInputs
        入力 user ((件数, 利用者の数)) と item ((件数, 品物の数)) の
        float32 (出す値の数は無し. 出力は全ての組の点数の表)
    """
    # 名前は学習のスクリプトと同梱の例 two_tower_retrieval の入力
    inputs = [
        DatasetInput(
            name=name, shape=[EXAMPLE_SIZE, size], dtype="float32",
            dataset_shape=[None, size]
        )
        for name, size in (("user", rank.users), ("item", rank.items))
    ]
    result = DatasetInputs(path=str(path), inputs=inputs, rank=rank)

    return result


@router.post("/dataset/inputs")
def read_dataset_inputs(body: DatasetBody) -> DatasetInputs:
    """
    データセットから, モデルの入力と出す値の数を読む関数

    問題の設定を渡し, それが表の生成なら, 入力の最後の次元と出す値の数を
    符号化した行の幅 (ML-Quest の table_encoding) にする (学習のスクリプト
    が学ぶ行の幅. 列の数とは違う). 並べるなら, 場所は反応の表のファイルで,
    入力 user と item の大きさを学習用の利用者と品物の数 (ML-Quest の
    rank_encoding) にする. 異常の検知のデータセットでは, 学習に使う列の数
    (ML-EDA の outputs) と, 問題の設定の度合いの決め方も返す

    Parameters
    ----------
    body: DatasetBody
        データセットのディレクトリ (並べるなら反応の表のファイル) と,
        問題の設定のファイル (任意)

    Returns
    ----------
    result: DatasetInputs
        入力 (形はバッチの次元を例の大きさにしたものと, データセットの
        形), タスク, 目的の列, クラス, 正解の種類, 出す値の数, 表の生成の
        符号化した行の幅, 並べるの学習用の利用者と品物の数, 異常の検知の
        学習に使う列の数と度合いの決め方

    Raises
    ----------
    HTTPException
        ML-EDA か ML-Quest が要るのに無い場合は 501, ディレクトリ, 説明か
        問題の設定が無い場合は 404, 読めない場合 (知らない版, 知らない項目の
        組など) と, 表の生成の幅か並べるの数を求められない場合と, 表の生成
        で入力が表の行 (バッチと列の 2 次元の 1 つ) でない場合と, 異常の
        検知の問題の設定でデータセットが異常の検知の表でない場合は 422.
        並べるで反応の表が無い場合は 404
    """
    # 並べるは反応の表のファイルから 2 塔の入力を作る
    path = absolute_path(body.path)
    problem = None
    if body.problem and body.problem.strip():
        problem = read_problem(absolute_path(body.problem))
        rank = rank_size(problem, path)
        if rank is not None:
            result = rank_inputs(path, rank)
            return result

    # 説明の版と読み方は ML-EDA に任せる
    if not path.is_dir():
        raise HTTPException(404, f"{path} にディレクトリがありません")
    eda = load_library("ml_eda", "model_inputs")
    try:
        info = eda.model_inputs(path)
    except FileNotFoundError as error:
        raise HTTPException(
            404,
            f"{path} は書き出したデータセットではありません "
            f"(dataset.yaml がありません): {error}"
        ) from error
    except READ_ERRORS as error:
        raise HTTPException(
            422, f"データセットを読めません: {error}"
        ) from error

    # 入力の名前, 形の例, 型と, 出す値の数 (形の崩れた値は 422)
    if not isinstance(info, dict) or not isinstance(info.get("inputs"), list):
        raise HTTPException(422, "データセットの入力を読めません")
    inputs = [
        DatasetInput(
            name=str(item["name"]), shape=example_shape(item.get("shape")),
            dtype=str(item.get("dtype") or "float32"),
            dataset_shape=list(item["shape"]),
            vocabulary=positive_int(item.get("vocabulary"))
        )
        for item in info["inputs"] if isinstance(item, dict)
        and "name" in item
    ]
    if not inputs:
        raise HTTPException(422, "データセットに入力がありません")
    outputs = info.get("outputs")
    classes = info.get("classes")
    task = optional_text(info, "task")
    target = optional_text(info, "target")

    # 異常の検知は, 入力と出力の幅を学習に使う列の数 (ML-EDA の outputs)
    # に. 問題の設定が異常の検知なら, データセットも異常の検知の表に限る
    score = None if problem is None else anomaly_score(problem)
    anomaly = None
    if score is not None and task != "anomaly":
        raise HTTPException(
            422,
            "異常の検知の問題の設定ですが, データセットが異常の検知の表 "
            "(task: anomaly) ではありません"
        )
    if task == "anomaly":
        anomaly = AnomalyWidth(
            columns=outputs if isinstance(outputs, int)
            and not isinstance(outputs, bool) and outputs > 0 else None,
            target=target, score=score or None,
            problem=None if score is None else problem_name(problem)
        )

    # 表の生成は, 入力 (バッチと列の 2 次元) の列の次元と出す値の数を
    # 符号化した行の幅に
    table = None if problem is None else table_width(problem, path)
    if table is not None:
        if len(inputs) != 1 or len(inputs[0].shape) != 2:
            raise HTTPException(
                422,
                "表の生成の問題の設定ですが, データセットの入力が表の行 "
                "(バッチと列の 2 次元の 1 つ) ではありません"
            )
        inputs = [inputs[0].model_copy(update={
            "shape": [EXAMPLE_SIZE, table.width],
            "dataset_shape": [None, table.width]
        })]
        outputs = table.width
    try:
        result = DatasetInputs(
            path=str(path), inputs=inputs, task=task, target=target,
            classes=list(classes) if isinstance(classes, list) else None,
            target_kind=optional_text(info, "target_kind"),
            outputs=outputs if isinstance(outputs, int)
            and not isinstance(outputs, bool) and outputs > 0 else None,
            table=table, anomaly=anomaly
        )
    except ValidationError as error:
        raise HTTPException(
            422, f"データセットの入力を読めません: {error}"
        ) from error

    return result


def optional_text(record: dict[str, Any], key: str) -> Optional[str]:
    """
    辞書の値を, 文字列か None にする関数

    Parameters
    ----------
    record: dict[str, Any]
        辞書
    key: str
        キー

    Returns
    ----------
    text: Optional[str]
        値の文字列 (無い場合は None. 数の列の名前なども文字列に)
    """
    value = record.get(key)
    text = None if value is None else str(value)

    return text


def section(record: Any, key: str) -> dict[str, Any]:
    """
    受け渡しの値の段階ごとの辞書を取り出す関数

    Parameters
    ----------
    record: Any
        ML-Quest の handoff の結果
    key: str
        段階の名前

    Returns
    ----------
    part: dict[str, Any]
        段階の値 (無いか辞書でない場合は空)
    """
    value = record.get(key) if isinstance(record, dict) else None
    part: dict[str, Any] = value if isinstance(value, dict) else {}

    return part


@functools.lru_cache(maxsize=1)
def example_names() -> frozenset[str]:
    """
    同梱の例の名前の一覧を返す関数 (同梱の例は起動の間に変わらないので
    1 度だけ読む)

    Parameters
    ----------
    None

    Returns
    ----------
    names: frozenset[str]
        ML-ModelControl の同梱の例の名前
    """
    names = frozenset(item["name"] for item in list_examples())

    return names


def known_example(value: Any) -> Optional[str]:
    """
    受け渡しの雛形の名前を, 同梱の例にあるものだけにする関数

    Parameters
    ----------
    value: Any
        ML-Quest の handoff の model.example

    Returns
    ----------
    example: Optional[str]
        同梱の例の名前 (無いか, 今の ML-ModelControl に無い名前なら None.
        開けない雛形を出さないため)
    """
    # 名前でない値は一覧を見ずに除く
    if not isinstance(value, str):
        example = None
        return example
    example = value if value in example_names() else None

    return example


def missing_record(problem_path: Path) -> str:
    """
    問題の設定にモデルの定義の保存先の記録が無い理由を返す関数

    Parameters
    ----------
    problem_path: Path
        問題の設定のファイル (from_problem が None を返したもの)

    Returns
    ----------
    reason: str
        プロジェクトを書き出していないか, 書き出したプロジェクトに定義の
        置き場の記録が無いか (異常の検知の距離に基づく度合いは, 定義が
        要らないこと)
    """
    # プロジェクトの場所の記録の有無で分ける (読み直しの間に読めなくなった
    # 場合は, 記録の無い文に)
    quest = load_library("ml_quest", "handoff")
    try:
        stages = quest.handoff(quest.load_problem(problem_path))
    except READ_ERRORS:
        stages = {}
    project_path = section(stages, "project").get("path")

    # 異常の検知の距離に基づく度合いは学習しないので, 定義が要らない
    if section(stages, "anomaly").get("score") == "distance":
        reason = (
            "度合いの決め方が距離に基づく度合い (学習しない) なので, モデルの"
            "定義は要りません"
        )
        return reason
    if project_path:
        reason = (
            f"プロジェクト {project_path} は書き出してありますが, モデルの"
            "定義の置き場の記録 (model.project_model_spec) がありません. "
            "プロジェクトのディレクトリを選ぶと, config の model.spec (無けれ"
            f"ば骨組みの既定の {DEFAULT_SPEC}) に保存します"
        )
        return reason
    reason = (
        "プロジェクトを書き出していません (ML-Pathfinder で書き出して"
        "ください. 書き出した後なら, ML-Quest が古いので "
        f"{SETUP_HINT} で入れ直す)"
    )

    return reason


def from_problem(problem_path: Path) -> Optional[ProjectTarget]:
    """
    問題の設定のプロジェクトの記録から, モデルの定義の保存先を求める関数

    Parameters
    ----------
    problem_path: Path
        問題の設定のファイル

    Returns
    ----------
    target: Optional[ProjectTarget]
        保存先 (記録が無い場合は None)

    Raises
    ----------
    HTTPException
        ML-Quest が無い場合は 501, 問題の設定を読めない場合と, 記録の
        保存先がプロジェクトの外か, プロジェクトの場所の記録が無い場合は
        422
    """
    # 読み方と版の確かめは ML-Quest に任せる
    quest = load_library("ml_quest", "handoff")
    try:
        problem = quest.load_problem(problem_path)
        stages = quest.handoff(problem)
    except READ_ERRORS as error:
        raise HTTPException(
            422, f"問題の設定を読めません: {error}"
        ) from error

    # 記録 (骨組みを書き出したときの保存先, プロジェクト, データセット)
    model = section(stages, "model")
    data = section(stages, "data")
    project = section(stages, "project")
    spec_path = model.get("project_model_spec")
    if not spec_path:
        target = None
        return target
    path = absolute_path(str(spec_path))
    project_path = project.get("path")
    if not project_path:
        raise HTTPException(
            422,
            "問題の設定にプロジェクトの場所 (project.path) の記録がありません"
        )
    root = absolute_path(str(project_path))
    if not (root / PROJECT_CONFIG).is_file():
        raise HTTPException(
            422,
            f"問題の設定が記録したプロジェクト {root} に {PROJECT_CONFIG} が"
            "ありません. プロジェクトを移したか消した場合は, ML-Pathfinder で"
            "書き出し直してください"
        )
    if not path.is_relative_to(root):
        raise HTTPException(
            422,
            f"問題の設定の保存先 {path} がプロジェクト {root} の外にあります"
        )
    dataset = data.get("project_dataset")
    name = problem.get("name") if isinstance(problem, dict) else None
    target = ProjectTarget(
        path=str(path), source="problem", exists=path.exists(),
        project=str(root),
        dataset=str(dataset) if dataset else None,
        problem=name if isinstance(name, str) else None,
        problem_path=str(problem_path),
        example=known_example(model.get("example"))
    )

    return target


def named_target(target: ProjectTarget, name: str) -> ProjectTarget:
    """
    既定の保存先を, プロジェクトの models/ の下の名前の保存先に替える関数

    モデルの候補を並べるため. config の model.spec は書き換えない (ML-
    Dashboard が候補ごとに書き換えて学習するため)

    Parameters
    ----------
    target: ProjectTarget
        既定の保存先 (プロジェクトの場所を持つ)
    name: str
        models/ の下の名前 (拡張子無し)

    Returns
    ----------
    named: ProjectTarget
        models/<名前>.yaml の保存先

    Raises
    ----------
    HTTPException
        名前が規則に合わない場合, プロジェクトの場所が無い場合と, models/
        かそのファイルがリンクの場合は 422
    """
    # 置き場の名前と同じ規則 (/ を受け取らないので models/ の直下だけ)
    if not NAME_PATTERN.fullmatch(name):
        raise HTTPException(
            422,
            f"名前 {name!r} は使えません. 英数字, _, - (先頭に - は使えない. "
            "64 文字まで. 拡張子は書かない) にしてください"
        )
    if target.project is None:
        raise HTTPException(
            422, "保存先のプロジェクトの場所が分かりません"
        )

    # 同じ名前の .yaml か .yml が既にあれば, そのファイル (大文字と小文字
    # だけ違うものは実際の名前) に書く. 候補に同じ名前を 2 つ並べないため
    folder = Path(target.project) / MODELS_DIR
    choices = [folder / f"{name}{suffix}" for suffix in SPEC_SUFFIXES]
    path = next(
        (item for item in choices if item.exists() or item.is_symlink()),
        choices[0]
    )
    path = actual_entry(folder, path)

    # リンクの先には書かない (models/ の外に出ないため)
    for item in (folder, path):
        if item.is_symlink():
            raise HTTPException(
                422,
                f"保存先 {item} はリンクです. 名前で選ぶ保存先はリンクに"
                "書きません"
            )
    named = target.model_copy(update={
        "path": str(path), "source": "name", "exists": path.exists(),
        "default": same_file(path, Path(target.path))
    })

    return named


def actual_entry(folder: Path, path: Path) -> Path:
    """
    既にあるファイルを, ディレクトリの中の実際の名前の場所にする関数

    大文字と小文字を区別しないファイルシステムで, 確かめの文と保存先に
    実際のファイル名を出すため

    Parameters
    ----------
    folder: Path
        models/ のディレクトリ
    path: Path
        名前から作った場所

    Returns
    ----------
    actual: Path
        同じファイルの実際の名前の場所 (無いか読めない場合はそのまま)
    """
    # 同じ名前があればそのまま, 無ければ大小を無視して同じファイルを探す
    actual = path
    if not path.exists():
        return actual
    try:
        children = list(folder.iterdir())
        if any(child.name == path.name for child in children):
            return actual
        for child in children:
            if child.name.casefold() == path.name.casefold() and (
                os.path.samefile(child, path)
            ):
                actual = child
                break
    except OSError:
        actual = path

    return actual


def same_file(first: Path, second: Path) -> bool:
    """
    2 つの場所が同じファイルを指すかを返す関数

    Parameters
    ----------
    first: Path
        1 つ目の場所
    second: Path
        2 つ目の場所

    Returns
    ----------
    same: bool
        同じ場所か, 両方あって同じファイルなら真
    """
    # 両方あればファイルで比べる
    same = first == second
    if same:
        return same
    if first.exists() and second.exists():
        try:
            same = os.path.samefile(first, second)
        except OSError:
            same = False
        return same

    # 無いファイルは場所の文字で比べ, 大文字と小文字だけ違う場合は置き場が
    # 区別しないかで決める (片方だけあるなら区別する置き場)
    if str(first).casefold() == str(second).casefold() and not (
        first.exists() or second.exists()
    ):
        same = ignores_case(first)

    return same


def ignores_case(path: Path) -> bool:
    """
    場所の置き場が大文字と小文字を区別しないかを返す関数

    まだ無いファイルが, 既にある別の名前のファイルと同じになるかを
    見分けるため. 一番近いある上のディレクトリの名前を, 大文字と小文字を
    入れ替えて探す

    Parameters
    ----------
    path: Path
        場所 (無くてよい)

    Returns
    ----------
    ignores: bool
        区別しないなら真. 確かめられる名前が無い場合も真 (同じファイルを
        既定の定義の上書きと知らせずに書かないため)
    """
    # 大文字と小文字を入れ替えて名前の変わる, ある上のディレクトリで試す
    for parent in (path, *path.parents):
        swapped = parent.name.swapcase()
        if swapped == parent.name or not parent.exists():
            continue
        other = parent.with_name(swapped)
        try:
            ignores = other.exists() and os.path.samefile(parent, other)
        except OSError:
            ignores = False
        return ignores
    ignores = True

    return ignores


def resolve_target(text: str, name: Optional[str] = None) -> ProjectTarget:
    """
    プロジェクトのディレクトリか問題の設定のファイルから, モデルの定義の
    保存先を決める関数

    Parameters
    ----------
    text: str
        プロジェクトのディレクトリか問題の設定のファイル
    name: Optional[str]
        models/ の下の名前 (無いか空なら既定の保存先)

    Returns
    ----------
    target: ProjectTarget
        保存先と, どこから決めたか

    Raises
    ----------
    HTTPException
        場所が無い場合は 404, 保存先を決められない場合は 422, ML-Quest
        が要るのに無い場合は 501
    """
    # 問題の設定のファイルなら記録が要る
    path = absolute_path(text)
    if path.is_file():
        found = from_problem(path)
        if found is None:
            raise HTTPException(
                422, f"{path.name} を選べません: {missing_record(path)}"
            )
        target = check_target(with_name(found, name))
        return target
    if not path.is_dir():
        raise HTTPException(404, f"{path} がありません")

    # プロジェクトのディレクトリは中の問題の設定の記録, 無ければ骨組みの既定
    found = None
    if (path / PROBLEM_FILE).is_file():
        found = from_problem(path / PROBLEM_FILE)
    if found is None:
        if not (path / PROJECT_CONFIG).is_file():
            raise HTTPException(
                422,
                f"{path} は学習のプロジェクト ({PROJECT_CONFIG} を持つ"
                "ディレクトリ) ではありません. プロジェクトのディレクトリか"
                "問題の設定のファイルを選んでください"
            )
        # config の model.spec (無ければ骨組みの既定). リンクをたどった先も
        # プロジェクトの中に限る
        spec_path = (path / config_spec(path)).resolve()
        if not spec_path.is_relative_to(path):
            raise HTTPException(
                422, f"保存先 {spec_path} がプロジェクト {path} の外にあります"
            )
        found = ProjectTarget(
            path=str(spec_path), source="project", exists=spec_path.exists(),
            project=str(path), dataset=config_dataset(path)
        )
    target = check_target(with_name(found, name))

    return target


def with_name(target: ProjectTarget, name: Optional[str]) -> ProjectTarget:
    """
    名前があれば models/ の下の保存先に, 無ければ既定のままにする関数

    Parameters
    ----------
    target: ProjectTarget
        既定の保存先
    name: Optional[str]
        models/ の下の名前 (無いか空なら既定)

    Returns
    ----------
    chosen: ProjectTarget
        使う保存先
    """
    # 空白だけの名前は既定
    text = (name or "").strip()
    chosen = named_target(target, text) if text else target

    return chosen


def config_dataset(project: Path) -> Optional[str]:
    """
    骨組みの config の data.dataset を, プロジェクトのディレクトリからの
    絶対パスにして返す関数

    問題の設定の記録の無いプロジェクトでも, 学習のスクリプトが読む
    データセットの場所を, 入力と出力を作る欄の候補に出すため

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    dataset: Optional[str]
        データセットの場所 (config を読めないか, 書いていなければ None)
    """
    # 候補に出すだけなので, 読めない config は黙って候補無し
    try:
        config = yaml.safe_load(
            (project / PROJECT_CONFIG).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        dataset = None
        return dataset
    value = section(config, "data").get("dataset")
    if not isinstance(value, str) or not value.strip():
        dataset = None
        return dataset
    dataset = str((project / value).resolve())

    return dataset


def config_spec(project: Path) -> Path:
    """
    骨組みの config の model.spec を返す関数

    学習のスクリプトが読むモデルの定義の場所に保存するため. 書いていない
    か読めない場合は骨組みの既定

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    spec: Path
        プロジェクトからのモデルの定義の場所
    """
    # 読めない config と書いていない場合は既定の models/model.yaml
    try:
        config = yaml.safe_load(
            (project / PROJECT_CONFIG).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        spec = DEFAULT_SPEC
        return spec
    value = section(config, "model").get("spec")
    spec = (Path(value) if isinstance(value, str) and value.strip()
            else DEFAULT_SPEC)

    return spec


def check_target(target: ProjectTarget) -> ProjectTarget:
    """
    保存先が書けるファイルの場所かを確かめる関数

    Parameters
    ----------
    target: ProjectTarget
        保存先

    Returns
    ----------
    target: ProjectTarget
        確かめた保存先 (そのまま)

    Raises
    ----------
    HTTPException
        拡張子が小文字の YAML でない場合, 学習の設定のディレクトリ (configs/) の中,
        プロジェクトの直下の problem.yaml, data/ か outputs/ の中の場合,
        途中の名前がディレクトリでない場合と, ディレクトリなどファイルで
        ないものがある場合は 422
    """
    # YAML のファイルだけに書く
    path = Path(target.path)
    if path.suffix.lower() not in SPEC_SUFFIXES:
        raise HTTPException(
            422, f"保存先 {path} は YAML のファイル (.yaml) ではありません"
        )

    # プロジェクトの中の他の段階のファイルは書き換えない
    if target.project is not None and path.is_relative_to(target.project):
        check_reserved(path, path.relative_to(target.project).parts)

    # 拡張子は小文字だけ (ML-Dashboard のモデルの定義の候補は小文字の拡張子
    # だけを並べるため)
    if path.suffix not in SPEC_SUFFIXES:
        raise HTTPException(
            422,
            f"保存先 {path} の拡張子が小文字ではありません. ML-Dashboard の"
            "モデルの定義の候補に出ないので, 小文字の .yaml か .yml にして"
            "ください (既にあるファイルは名前を直す)"
        )

    # 途中の名前がファイルなどなら, 保存でディレクトリを作れない (壊れた
    # リンクも同じ)
    for parent in path.parents:
        if parent.is_dir():
            continue
        if parent.exists() or parent.is_symlink():
            raise HTTPException(
                422,
                f"保存先 {path} の途中の {parent} がディレクトリではありま"
                "せん"
            )
    if path.exists() and not path.is_file():
        raise HTTPException(
            422, f"保存先 {path} はファイルではありません"
        )

    return target


def check_reserved(path: Path, parts: tuple[str, ...]) -> None:
    """
    保存先が, プロジェクトの中の他の段階のファイルの場所でないかを確かめる
    関数

    学習の設定 (configs/ の下), 問題の設定の写し (直下の problem.yaml),
    データと実験 (data/ と outputs/ の下) をモデルの定義で上書きしない
    ため. 大文字と小文字を区別しないファイルシステムでも同じ場所を断る
    ため, 名前は大小を無視して比べる

    Parameters
    ----------
    path: Path
        保存先
    parts: tuple[str, ...]
        プロジェクトのディレクトリからの保存先の名前の並び

    Returns
    ----------
    None

    Raises
    ----------
    HTTPException
        他の段階のファイルの場所なら 422
    """
    # 直下の名前で見分ける
    head = parts[0].casefold() if parts else ""
    if head == CONFIG_DIR:
        raise HTTPException(
            422,
            f"保存先 {path} は学習の設定のディレクトリ ({CONFIG_DIR}/) "
            "の中です. モデルの定義は models/ などに置いてください"
        )
    if len(parts) == 1 and head == PROBLEM_FILE:
        raise HTTPException(
            422,
            f"保存先 {path} はプロジェクトの問題の設定 ({PROBLEM_FILE}) "
            "です. モデルの定義は models/ などに置いてください"
        )
    if len(parts) > 1 and head in RESERVED_DIRS:
        raise HTTPException(
            422,
            f"保存先 {path} は{RESERVED_DIRS[head]} ({head}/) の中です. "
            "モデルの定義は models/ などに置いてください"
        )


def make_parents(path: Path, project: Optional[str]) -> None:
    """
    保存先までの無いディレクトリを, プロジェクトの中だけ作る関数

    消したプロジェクトの場所を作り直したり, プロジェクトの外にディレクトリ
    を作ったりしないため

    Parameters
    ----------
    path: Path
        保存先
    project: Optional[str]
        プロジェクトのディレクトリ

    Returns
    ----------
    None

    Raises
    ----------
    HTTPException
        プロジェクトが無い場合と, 保存先がプロジェクトの外の場合は 422
    OSError
        ディレクトリを作れない場合
    """
    # プロジェクトそのものは作らない
    root = Path(project) if project is not None else None
    if root is None or not root.is_dir():
        raise HTTPException(
            422, f"保存先のプロジェクトのディレクトリ {root} がありません"
        )
    if not path.parent.is_relative_to(root):
        raise HTTPException(
            422, f"保存先 {path} がプロジェクト {root} の外にあります"
        )

    # プロジェクトの直下から順に, 無いものだけ作る
    missing = [
        parent for parent in path.parents
        if parent.is_relative_to(root) and parent != root
        and not parent.is_dir()
    ]
    for parent in reversed(missing):
        parent.mkdir(exist_ok=True)

    # 作った後のリンクの先もプロジェクトの中に限る
    if not path.parent.resolve().is_relative_to(root):
        raise HTTPException(
            422, f"保存先 {path} がプロジェクト {root} の外にあります"
        )


def modified_at(path: Path) -> Optional[str]:
    """
    ファイルの更新時刻を ISO 8601 の文字にする関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    stamp: Optional[str]
        手元の時刻帯の更新時刻 (読めなければ None)
    """
    # 一覧の並べ替えと表示の時刻 (ML-Fathom の一覧と同じ)
    try:
        moment = dt.datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        stamp = None
        return stamp
    stamp = moment.astimezone().isoformat(timespec="seconds")

    return stamp


def problem_entry(path: Path) -> ProblemEntry:
    """
    ML-Pathfinder の置き場の問題の設定を, 一覧の 1 件にする関数

    Parameters
    ----------
    path: Path
        problem.yaml

    Returns
    ----------
    entry: ProblemEntry
        場所, 名前, 時刻, 使えるか (使えない理由), プロジェクトとデータ
        セットの場所, 保存先
    """
    # 読めないものも理由を付けて並べる (名前はディレクトリの名前)
    entry = ProblemEntry(
        path=str(path), name=path.parent.name,
        updated_at=modified_at(path), usable=False
    )
    quest = load_library("ml_quest", "handoff")
    try:
        problem = quest.load_problem(path)
    except READ_ERRORS as error:
        entry.reason = f"問題の設定を読めません: {error}"
        return entry
    name = problem.get("name") if isinstance(problem, dict) else None
    created = problem.get("created_at") if isinstance(problem, dict) else None
    if not isinstance(name, str) or not created:
        entry.reason = f"{path.name} は問題の設定のファイルではありません"
        return entry
    entry.name = name
    entry.created_at = str(created)

    # 保存先は「プロジェクトに保存」で確かめるときと同じ決め方
    try:
        found = from_problem(path)
        if found is None:
            entry.reason = missing_record(path)
            return entry
        target = check_target(found)
    except HTTPException as error:
        entry.reason = str(error.detail)
        return entry
    entry.usable = True
    entry.project = target.project
    entry.dataset = target.dataset
    entry.target = target.path
    entry.example = target.example

    return entry


def updated_of(entry: ProblemEntry) -> str:
    """
    一覧の 1 件の更新時刻を, 並べ替えの鍵にする関数

    Parameters
    ----------
    entry: ProblemEntry
        一覧の 1 件

    Returns
    ----------
    key: str
        更新時刻の文字 (無ければ空)
    """
    # 時刻の無いものは後ろ
    key = entry.updated_at or ""

    return key


@router.get("/problems")
def list_problems() -> ProblemList:
    """
    ML-Pathfinder の置き場にある問題の設定の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    result: ProblemList
        置き場の problems, 問題の設定 (更新の新しい順), 置き場を読めない
        場合の理由

    Raises
    ----------
    HTTPException
        ML-Quest が無い場合は 501
    """
    # problems/<識別子>/problem.yaml (置き場が無ければ空. ML-Fathom と同じ)
    load_library("ml_quest", "handoff")
    root = get_problems_dir()
    files: list[Path] = []
    error = None
    try:
        if root.is_dir():
            files = [
                child / PROBLEM_FILE for child in sorted(root.iterdir())
                if (child / PROBLEM_FILE).is_file()
            ]
    except OSError as reason:
        error = f"{root} を読めません: {reason}"
    problems = [problem_entry(file) for file in files]
    problems.sort(key=updated_of, reverse=True)
    result = ProblemList(root=str(root), problems=problems, error=error)

    return result


@router.post("/project/target")
def project_target(body: TargetBody) -> ProjectTarget:
    """
    プロジェクトのモデルの定義の保存先を返す関数

    Parameters
    ----------
    body: TargetBody
        プロジェクトのディレクトリか問題の設定のファイルと, models/ の下の
        名前

    Returns
    ----------
    target: ProjectTarget
        保存先, プロジェクトとデータセットの場所, 既にあるか
    """
    target = resolve_target(body.path, body.name)

    return target


@router.put("/project/model")
def save_project_model(body: ProjectSaveBody) -> ProjectSaved:
    """
    定義をプロジェクトの保存先に書く関数

    Parameters
    ----------
    body: ProjectSaveBody
        保存先を決める場所, models/ の下の名前, 定義, 上書きするか

    Returns
    ----------
    saved: ProjectSaved
        書いたファイルと更新時刻

    Raises
    ----------
    HTTPException
        確かめた保存先 (target) と違う場合と, 既にあり上書きしない場合は
        409, 定義を読めない場合は 422, 書けない場合は 500 (他は保存先を
        決めるときと同じ)
    """
    # 押した時点の保存先を決め直し, 確かめた保存先と違うか, 既にあれば
    # 確認を求める (上書きの旗を確かめた保存先にだけ効かせるため)
    target = resolve_target(body.path, body.name)
    path = Path(target.path)
    if body.target is not None and body.target != target.path:
        raise HTTPException(
            409,
            f"保存先が, 確かめた {body.target} から {path} に変わりました"
        )
    if path.exists() and not body.overwrite:
        raise HTTPException(409, f"{path} が既にあります")
    try:
        text = spec_text(body.spec)
    except ModelError as error:
        raise HTTPException(422, str(error)) from error

    # 保存先までの無いディレクトリは, プロジェクトの中だけ作る
    try:
        make_parents(path, target.project)
        updated_at = write_atomic(path, text)
    except OSError as error:
        raise HTTPException(
            500, f"{path} に書けません: {error}"
        ) from error
    saved = ProjectSaved(path=str(path), updated_at=updated_at)

    return saved
