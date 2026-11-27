"""
データセットとプロジェクトにつなぐ API

データセットの説明は ML-EDA, 問題の設定は ML-Quest を通して読む (他の段階
のファイルは, その約束を持つライブラリを通して読む決まり). どちらも追加の
依存 project で入る
"""

import copy
import datetime as dt
import functools
import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException, Request
from ml_modelcontrol import infer, list_examples
from pydantic import BaseModel, Field, ValidationError

from ml_createarchitecture.store import (
    NAME_PATTERN,
    ModelError,
    spec_text,
    write_atomic,
)

router = APIRouter()

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

# 出力を (件数,) か (件数, 数) に reshape して読む骨組みの型 (出力の大きさ
# をバッチの次元を除いた積で比べる. 他の骨組みは出力をそのまま使うので,
# (件数, 数) の 2 次元で比べる)
RESHAPED_SKELETONS = frozenset({"uplift", "price", "demand"})

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
        "異常の検知なら, 度合いの決め方を受け持ちの答えから読む. 効果を"
        "見積もるなら, データセットの説明に無い処置の列と行の id の列を受け"
        "持ちの答えから読む. 値段を変える判断なら, データセットの説明に無い"
        "値段, 原価, 商品の id と行の id の列を受け持ちの答えから読む. 時間を"
        "またぐ量の判断なら, 出す値の数を学ぶ分位点の水準の数 (記録の"
        "プロジェクトの学習の設定の demand.levels. 無ければ受け渡しの "
        "demand.levels) にする)"
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
        "列と行の id の列は数えず, one_hot で増えた列は数える)"
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


class UpliftWidth(BaseModel):
    # 効果を見積もるの入力の幅 (処置の列を除いた学習に使う列の数) と,
    # 数えない列
    columns: Optional[int] = Field(
        description="学習に使う列の数 (入力 x の幅. 処置の列, 結果の列と"
        "行の id の列は数えず, one_hot で増えた列は数える)"
    )
    outcome: Optional[str] = Field(
        default=None, description="結果の列 (データセットの目的の列)"
    )
    treatment: Optional[str] = Field(
        default=None,
        description="処置の列 (データセットの説明の uplift.treatment. 無け"
        "れば問題の設定の答え. どちらにも無ければ無し)"
    )
    id: Optional[str] = Field(
        default=None,
        description="行の id の列 (データセットの説明の uplift.id. 無けれ"
        "ば問題の設定の答え. どちらにも無ければ無し)"
    )
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )


class PriceWidth(BaseModel):
    # 値段を変える判断の入力の幅 (商品の特徴の列と値段の列の数) と, 列の
    # 名前
    columns: Optional[int] = Field(
        description="特徴の列と値段の列の数 (入力 x の幅. 値段の列は最後. "
        "売れた数, 元の値段, 原価, 商品の id と行の id の列は数えず, "
        "one_hot で増えた列は数える)"
    )
    price: Optional[str] = Field(
        default=None,
        description="値段の列 (データセットの説明の price.price. 説明に無け"
        "れば問題の設定の答え. どちらにも無ければ無し)"
    )
    units: Optional[str] = Field(
        default=None, description="売れた数の列 (データセットの目的の列)"
    )
    cost: Optional[str] = Field(
        default=None,
        description="原価の列 (データセットの説明の price.cost. 説明に項目"
        "が無ければ問題の設定の答え. 説明が null か, どちらにも無ければ無し)"
    )
    product: Optional[str] = Field(
        default=None,
        description="商品の id の列 (データセットの説明の price.product. "
        "説明に項目が無ければ問題の設定の答え. 説明が null か, どちらにも"
        "無ければ無し)"
    )
    id: Optional[str] = Field(
        default=None,
        description="行の id の列 (データセットの説明の price.id. 説明に"
        "無ければ問題の設定の答え. どちらにも無ければ無し)"
    )
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )


class DemandWidth(BaseModel):
    # 時間をまたぐ量の判断の入力の幅 (学習に使う列の数), 数えない列と,
    # 出力 quantiles の値の数にする学ぶ分位点の水準
    columns: Optional[int] = Field(
        description="学習に使う列の数 (入力 x の幅. 日ごとの売れた数, 先の"
        "期間と前の期間の合計, 商品の id, 日付, 在庫の数と行の id の列は"
        "数えず, one_hot で増えた列は数える)"
    )
    units: Optional[str] = Field(
        default=None,
        description="日ごとの売れた数の列 (データセットの説明の "
        "demand.units. 説明に無ければ問題の設定の答え. どちらにも無ければ"
        "無し)"
    )
    product: Optional[str] = Field(
        default=None,
        description="商品の id の列 (データセットの説明の demand.product. "
        "説明に無ければ問題の設定の答え. どちらにも無ければ無し)"
    )
    date: Optional[str] = Field(
        default=None,
        description="日付の列 (データセットの説明の demand.date. 説明に"
        "無ければ問題の設定の答え. どちらにも無ければ無し)"
    )
    stock: Optional[str] = Field(
        default=None,
        description="在庫の数の列 (データセットの説明の demand.stock. 説明"
        "に項目が無ければ問題の設定の答え. 説明が null か, どちらにも無け"
        "れば無し)"
    )
    id: Optional[str] = Field(
        default=None,
        description="行の id の列 (データセットの説明の demand.id. 説明に"
        "無ければ問題の設定の答え. どちらにも無ければ無し)"
    )
    levels: Optional[list[float]] = Field(
        default=None,
        description="学ぶ分位点の水準 (出力 quantiles の列の順. 記録の"
        "プロジェクトの学習の設定の demand.levels を先, 無ければ問題の設定の"
        "受け渡しの demand.levels. 時間をまたぐ量の判断の問題の設定を渡して"
        "いないか, どちらも読めなければ無し)"
    )
    levels_from: Optional[str] = Field(
        default=None,
        description="水準をどこから読んだか (config は学習の設定, problem "
        "は問題の設定. 水準が無ければ無し)"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="水準の決め方の知らせ (学習の設定と問題の設定の水準の数"
        "の食い違い, 学習の設定や問題の設定を読めないことなど)"
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
    uplift: Optional[UpliftWidth] = Field(
        default=None,
        description="効果を見積もるのデータセットか問題の設定で, 入力 x の幅"
        "にした学習に使う列の数と数えない列 (他は無し)"
    )
    price: Optional[PriceWidth] = Field(
        default=None,
        description="値段を変える判断のデータセットか問題の設定で, 入力 x "
        "の幅にした特徴の列と値段の列の数と列の名前 (他は無し)"
    )
    demand: Optional[DemandWidth] = Field(
        default=None,
        description="需要の記録のデータセットか時間をまたぐ量の判断の問題の"
        "設定で, 入力 x の幅にした学習に使う列の数, 数えない列と, 出力の"
        "値の数にした学ぶ分位点の水準 (他は無し)"
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
    accept_contract: bool = Field(
        default=False,
        description="骨組みの約束に合わない所があっても書くか (偽なら, 合わ"
        "ない所を detail の findings に付けて 409)"
    )


class ProjectSaved(BaseModel):
    # 書いたファイル, 更新時刻と, 骨組みの約束の確かめの知らせ
    path: str
    updated_at: str
    notes: list[str] = Field(
        default_factory=list,
        description="骨組みの約束の確かめの知らせ (データセットを読めず, "
        "入力と出力の形の大きさを確かめていない, 出力の形を推定でき"
        "ないなど)"
    )


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


def project_config(problem: Any) -> tuple[bool, Optional[dict[str, Any]]]:
    """
    問題の設定が記録したプロジェクトの学習の設定 (config) を読む関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    found: bool
        記録のプロジェクトに config のファイルがあるか
    config: Optional[dict[str, Any]]
        config の中身 (無いか, 読めないか, 辞書でなければ None)
    """
    # プロジェクトの場所は ML-Quest の記録の読み方に任せる
    quest = load_library("ml_quest", "project_of")
    try:
        place = quest.project_of(problem).get("path")
    except READ_ERRORS:
        place = None
    if not isinstance(place, str) or not (
        Path(place) / PROJECT_CONFIG
    ).is_file():
        return False, None
    try:
        loaded = yaml.safe_load(
            (Path(place) / PROJECT_CONFIG).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        loaded = None
    config = loaded if isinstance(loaded, dict) else None

    return True, config


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
    found, config = project_config(problem)
    rates: dict[str, float] = {}
    notes: list[str] = []
    if not found:
        return rates, notes
    if config is None:
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
    # ML-Quest の答えの読み方 (ML-Quest の異常の検知の受け持ちと同じ
    # 見分け方)
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


def logged_answers(
    problem: Any, keys: tuple[str, ...]
) -> Optional[dict[str, Any]]:
    """
    問題の設定の受け持ちが今ある記録から学ぶ効果を見積もるなら, 受け持ちの
    答えを返す関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)
    keys: tuple[str, ...]
        受け持ちの中の答えの場所 (varied は常に読む)

    Returns
    ----------
    values: Optional[dict[str, Any]]
        場所から答え (答えていなければ None. varied も含む). 効果を見積もる
        で方策でなく, 今ある記録から集め, 作り方が既製のモデルをそのまま
        でない受け持ちでなければ None. 要らなくなった問いの古い答えは除外
    """
    # ML-Quest の答えの読み方 (ML-Quest の今ある記録から学ぶ受け持ちと同じ
    # 見分け方)
    quest = load_library("ml_quest", "active_value")
    try:
        values = {
            key: quest.active_value(problem, f"{quest.PART}.{key}")
            for key in (
                "role", "autonomous", "collect", "method", "varied", *keys
            )
        }
    except READ_ERRORS:
        values = {}
    if values.get("role") != "effect" or values.get("autonomous") is True \
            or values.get("collect") != "log" \
            or values.get("method") == "pretrained":
        return None

    return values


def column_names(
    values: dict[str, Any], keys: dict[str, str]
) -> dict[str, Optional[str]]:
    """
    答えか説明の値から, 列の名前 (空でない文字列) を取り出す関数

    Parameters
    ----------
    values: dict[str, Any]
        答えか説明の値
    keys: dict[str, str]
        返す名前から values の場所

    Returns
    ----------
    columns: dict[str, Optional[str]]
        返す名前から列の名前 (無いか文字列でなければ None)
    """
    # 空の文字列と文字列でない値は無し
    columns: dict[str, Optional[str]] = {}
    for name, key in keys.items():
        value = values.get(key)
        columns[name] = value if isinstance(value, str) and value else None

    return columns


def uplift_columns(problem: Any) -> Optional[dict[str, Optional[str]]]:
    """
    問題の設定の受け持ちが効果の見積もりなら, 処置の列と行の id の列を返す
    関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    columns: Optional[dict[str, Optional[str]]]
        treatment (処置の列) と id (行の id の列) の答え (答えていなければ
        None). 今ある記録から学ぶ効果を見積もるで, 記録で変えたものが値段で
        ない受け持ちでなければ None
    """
    # 記録で変えたものが値段なら値段を変える判断 (ML-Quest の効果の見積もり
    # の受け持ちと同じ見分け方)
    values = logged_answers(problem, ("uplift.treatment", "uplift.id"))
    if values is None or values.get("varied") == "price":
        return None
    columns = column_names(
        values, {"treatment": "uplift.treatment", "id": "uplift.id"}
    )

    return columns


def price_columns(problem: Any) -> Optional[dict[str, Optional[str]]]:
    """
    問題の設定の受け持ちが値段を変える判断なら, 値段, 原価, 商品の id と
    行の id の列を返す関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    columns: Optional[dict[str, Optional[str]]]
        price (値段の列), cost (原価の列), product (商品の id の列) と id
        (行の id の列) の答え (答えていなければ None). 今ある記録から学ぶ
        効果を見積もるで, 記録で変えたものが値段の受け持ちでなければ None
    """
    # 記録で変えたものが値段 (ML-Quest の値段を変える判断の受け持ちと同じ
    # 見分け方)
    keys = {
        name: f"price.{name}" for name in ("price", "cost", "product", "id")
    }
    values = logged_answers(problem, tuple(keys.values()))
    if values is None or values.get("varied") != "price":
        return None
    columns = column_names(values, keys)

    return columns


def dataset_section(path: Path, key: str) -> dict[str, Any]:
    """
    書き出したデータセットの説明から, 1 つの項目を読む関数

    Parameters
    ----------
    path: Path
        書き出したデータセットのディレクトリ
    key: str
        説明の項目の名前 (uplift, price など)

    Returns
    ----------
    part: dict[str, Any]
        説明の項目 (無いか, 辞書でないか, 読めなければ空)
    """
    # ML-EDA の説明の読み方
    eda = load_library("ml_eda", "load_dataset")
    try:
        info = eda.load_dataset(path).info
    except READ_ERRORS:
        info = {}
    part = section(info, key)

    return part


def dataset_uplift(path: Path) -> dict[str, Optional[str]]:
    """
    介入の記録の表の説明から, 処置の列と行の id の列を読む関数

    Parameters
    ----------
    path: Path
        書き出したデータセットのディレクトリ (task: uplift)

    Returns
    ----------
    columns: dict[str, Optional[str]]
        treatment (処置の列) と id (行の id の列). 説明に無いか読めなければ
        None
    """
    # 説明の uplift の名前
    columns = column_names(
        dataset_section(path, "uplift"),
        {"treatment": "treatment", "id": "id"}
    )

    return columns


def described_names(
    written: dict[str, Any], answers: Optional[dict[str, Any]],
    keys: tuple[str, ...]
) -> dict[str, Optional[str]]:
    """
    列の名前を, データセットの説明を先に, 説明に項目が無ければ問題の設定の
    答えから決める関数

    Parameters
    ----------
    written: dict[str, Any]
        データセットの説明の項目 (price, demand など. 無ければ空)
    answers: Optional[dict[str, Any]]
        問題の設定の答え (その受け持ちの問題の設定でなければ None)
    keys: tuple[str, ...]
        列の名前の場所 (説明と答えで同じ名前)

    Returns
    ----------
    columns: dict[str, Optional[str]]
        場所から列の名前 (無ければ None. 説明の値が null の項目は, その列
        が無いので答えで埋めない)
    """
    # 説明にある項目の説明の値 (null は列が無い印) と, 説明に無い項目の
    # 問題の設定の答え
    columns = {
        key: (
            column_names(written, {key: key})[key] if key in written
            else column_names(answers or {}, {key: key})[key]
        )
        for key in keys
    }

    return columns


def price_names(
    written: dict[str, Any], answers: Optional[dict[str, Optional[str]]]
) -> dict[str, Optional[str]]:
    """
    値段, 原価, 商品の id と行の id の列の名前を, データセットの説明を先に,
    説明に項目が無ければ問題の設定の答えから決める関数

    Parameters
    ----------
    written: dict[str, Any]
        データセットの説明の price (無ければ空)
    answers: Optional[dict[str, Optional[str]]]
        問題の設定の答え (price_columns. 値段を変える判断の問題の設定で
        なければ None)

    Returns
    ----------
    columns: dict[str, Optional[str]]
        price, cost, product, id の列の名前 (無ければ None. 説明の値が null
        の項目は, その列が無いので答えで埋めない)
    """
    columns = described_names(
        written, answers, ("price", "cost", "product", "id")
    )

    return columns


def demand_answers(problem: Any) -> Optional[dict[str, Any]]:
    """
    問題の設定の受け持ちが時間をまたぐ量の判断なら, 列の答えと学ぶ分位点の
    水準を返す関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    answers: Optional[dict[str, Any]]
        product, date, units, stock, id (列の答え. 答えていなければ None),
        levels (受け渡しの demand.levels. 受け渡しにできなければ None) と
        error (受け渡しにできない理由. できれば None). 予測するで正解の
        種類が先の期間の需要, 作り方が既製のモデルをそのままでない受け
        持ちでなければ None. 要らなくなった問いの古い答えは除外
    """
    # 受け持ちの見分け方は ML-Quest の時間をまたぐ量の判断の受け持ちと同じ
    quest = load_library("ml_quest", "active_value")
    names = ("product", "date", "units", "stock", "id")
    try:
        role = quest.active_value(problem, f"{quest.PART}.role")
        kind = quest.active_value(problem, f"{quest.PART}.target.kind")
        method = quest.active_value(problem, f"{quest.PART}.method")
        answers: Optional[dict[str, Any]] = {
            name: quest.active_value(problem, f"{quest.PART}.demand.{name}")
            for name in names
        }
    except READ_ERRORS:
        role = kind = method = None
        answers = None
    if role != "predict" or kind != "demand" or method == "pretrained" \
            or answers is None:
        return None

    # 学ぶ水準は ML-Quest の受け渡しが決める (答えた水準に 0.5 と水準 τ を
    # 足したもの. 骨組みの config の demand.levels と同じ)
    answers.update(levels=None, error=None)
    try:
        levels = section(quest.handoff(problem), "demand").get("levels")
    except READ_ERRORS as error:
        answers["error"] = str(error)
        return answers
    if isinstance(levels, list) and levels and all(
        isinstance(level, (int, float)) and not isinstance(level, bool)
        for level in levels
    ):
        answers["levels"] = [float(level) for level in levels]

    return answers


def config_levels(config: dict[str, Any]) -> Optional[list[float]]:
    """
    学習の設定の demand.levels を読む関数

    Parameters
    ----------
    config: dict[str, Any]
        学習の設定 (project_config)

    Returns
    ----------
    levels: Optional[list[float]]
        水準の並び (無いか, 空か, 0 より大きく 1 より小さい数を小さい順に
        重ねずに並べ, 0.5 を含む並びでなければ None)
    """
    # 骨組みの学習のスクリプトが止める値は使わない (ML-Quest の時間をまたぐ
    # 量の判断の骨組みの config の確かめと同じ条件)
    value = section(config, "demand").get("levels")
    if not isinstance(value, list) or not value or not all(
        isinstance(level, (int, float)) and not isinstance(level, bool)
        and 0 < level < 1 for level in value
    ) or any(
        first >= second
        for first, second in zip(value, value[1:], strict=False)
    ) or 0.5 not in value:
        return None
    levels = [float(level) for level in value]

    return levels


def demand_levels(
    problem: Any, answers: dict[str, Any]
) -> tuple[Optional[list[float]], Optional[str], list[str]]:
    """
    時間をまたぐ量の判断の学ぶ分位点の水準を, 記録のプロジェクトの学習の
    設定を先に, 無ければ問題の設定の受け渡しから決める関数

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem. 時間をまたぐ量の判断)
    answers: dict[str, Any]
        問題の設定の答えと受け渡しの水準 (demand_answers)

    Returns
    ----------
    levels: Optional[list[float]]
        学ぶ水準 (どちらも読めなければ None)
    origin: Optional[str]
        config (学習の設定) か problem (問題の設定) (水準が無ければ None)
    notes: list[str]
        決め方の知らせ (水準の数の食い違い, 読めない学習の設定や問題の
        設定)
    """
    # 骨組みの学習のスクリプトが読むのは学習の設定の demand.levels なので,
    # 書いてあればそれを使い, 問題の設定と数が違えば知らせる
    notes: list[str] = []
    found, config = project_config(problem)
    written = None if config is None else config_levels(config)
    asked = answers.get("levels")
    if written is not None:
        if asked is not None and len(asked) != len(written):
            notes.append(
                f"学習の設定 ({PROJECT_CONFIG}) の学ぶ水準の数 "
                f"{len(written)} は, 問題の設定の水準の数 {len(asked)} と"
                "違います. 学習のスクリプトが読む学習の設定の水準の数にし"
                "ました (問題の設定に合わせるなら, 学習の設定の "
                "demand.levels を直してください)"
            )
        return written, "config", notes
    if found:
        notes.append(
            f"学習の設定 ({PROJECT_CONFIG}) の学ぶ水準 (demand.levels) を"
            "読めないので, 問題の設定の水準にしました (学習のスクリプトは, "
            "0 より大きく 1 より小さい数を小さい順に重ねずに並べ, 0.5 を含む"
            "水準でなければ止まるので, 学習の設定を直してください)"
        )
    if asked is None:
        reason = answers.get("error")
        notes.append(
            "問題の設定の学ぶ水準を読めません"
            + (f" ({reason})" if reason else "")
        )
        return None, None, notes

    return asked, "problem", notes


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
    (ML-EDA の outputs) と, 問題の設定の度合いの決め方も返す. 効果を
    見積もるのデータセットでは, 学習に使う列の数 (ML-EDA の入力 x の幅) と,
    処置の列と行の id の列 (データセットの説明を先, 無ければ問題の設定の
    答え) も返す. 値段を変える判断のデータセットでは, 特徴の列と値段の列の
    数 (ML-EDA の入力 x の幅) と, 値段, 売れた数, 原価, 商品の id と行の id
    の列 (データセットの説明を先, 説明に無ければ問題の設定の答え) も返す.
    需要の記録のデータセットでは, 学習に使う列の数 (ML-EDA の入力 x の
    幅), 数えない列 (データセットの説明を先, 説明に無ければ問題の設定の
    答え) と, 時間をまたぐ量の判断の問題の設定なら学ぶ分位点の水準 (記録の
    プロジェクトの学習の設定を先, 無ければ問題の設定の受け渡し) も返し,
    出す値の数を水準の数にする (水準が分からなければ無し)

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
        学習に使う列の数と度合いの決め方, 効果を見積もるの学習に使う列の数と
        数えない列, 値段を変える判断の特徴の列と値段の列の数と列の名前,
        需要の学習に使う列の数, 数えない列と学ぶ分位点の水準

    Raises
    ----------
    HTTPException
        ML-EDA か ML-Quest が要るのに無い場合は 501, ディレクトリ, 説明か
        問題の設定が無い場合は 404, 読めない場合 (知らない版, 知らない項目の
        組など) と, 表の生成の幅か並べるの数を求められない場合と, 表の生成
        で入力が表の行 (バッチと列の 2 次元の 1 つ) でない場合と, 異常の
        検知の問題の設定でデータセットが異常の検知の表でない場合と, 効果を
        見積もるの問題の設定でデータセットが介入の記録の表でない場合と, 値段
        を変える判断の問題の設定でデータセットが値段を変えた記録の表でない
        場合と, 時間をまたぐ量の判断の問題の設定でデータセットが需要の記録の
        表でない場合は 422.
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

    # 異常の検知の入力と出力の幅 (ML-EDA の outputs) と, 問題の設定が
    # 異常の検知の場合の表の種類の確かめ
    score = None if problem is None else anomaly_score(problem)
    anomaly = None
    if score is not None and task != "anomaly":
        raise HTTPException(
            422,
            "異常の検知の問題の設定ですが, データセットが ML-Fathom (ML-EDA) "
            "で異常の検知として書き出した表ではありません"
        )
    if task == "anomaly":
        anomaly = AnomalyWidth(
            columns=outputs if isinstance(outputs, int)
            and not isinstance(outputs, bool) and outputs > 0 else None,
            target=target, score=score or None,
            problem=None if score is None else problem_name(problem)
        )

    # 効果を見積もるの入力 x の幅 (ML-EDA の入力 x の最後の次元), 数えない
    # 列の名前 (データセットの説明を先, 無ければ問題の設定の答え) と, 問題
    # の設定が効果の見積もりの場合の表の種類の確かめ
    roles = None if problem is None else uplift_columns(problem)
    uplift = None
    if roles is not None and task != "uplift":
        raise HTTPException(
            422,
            "効果を見積もるの問題の設定ですが, データセットが ML-Fathom "
            "(ML-EDA) で介入の記録の表として書き出したものではありません"
        )
    if task == "uplift":
        last = inputs[0].dataset_shape[-1:] if len(inputs) == 1 else []
        written = dataset_uplift(path)
        uplift = UpliftWidth(
            columns=last[0] if last and isinstance(last[0], int)
            and last[0] > 0 else None,
            outcome=target,
            treatment=written["treatment"]
            or (roles or {}).get("treatment"),
            id=written["id"] or (roles or {}).get("id"),
            problem=None if roles is None else problem_name(problem)
        )

    # 値段を変える判断の入力 x の幅 (ML-EDA の入力 x の最後の次元), 列の
    # 名前 (データセットの説明を先, 説明に無ければ問題の設定の答え) と, 問題
    # の設定が値段を変える判断の場合の表の種類の確かめ
    answers = None if problem is None else price_columns(problem)
    price = None
    if answers is not None and task != "price":
        raise HTTPException(
            422,
            "値段を変える判断の問題の設定ですが, データセットが ML-Fathom "
            "(ML-EDA) で値段を変えた記録の表として書き出したものではありま"
            "せん"
        )
    if task == "price":
        last = inputs[0].dataset_shape[-1:] if len(inputs) == 1 else []
        names = price_names(dataset_section(path, "price"), answers)
        price = PriceWidth(
            columns=last[0] if last and isinstance(last[0], int)
            and last[0] > 0 else None,
            units=target, **names,
            problem=None if answers is None else problem_name(problem)
        )

    # 時間をまたぐ量の判断の入力 x の幅 (ML-EDA の入力 x の最後の次元),
    # 数えない列の名前 (データセットの説明を先, 説明に無ければ問題の設定の
    # 答え), 出す値の数にする学ぶ分位点の水準の数 (ML-EDA の outputs は
    # 水準 1 つあたりの 1) と, 問題の設定が時間をまたぐ量の判断の場合の表の
    # 種類の確かめ
    needs = None if problem is None else demand_answers(problem)
    demand = None
    if needs is not None and task != "demand":
        raise HTTPException(
            422,
            "時間をまたぐ量の判断の問題の設定ですが, データセットが "
            "ML-Fathom (ML-EDA) で需要の記録の表として書き出したものでは"
            "ありません"
        )
    if task == "demand":
        last = inputs[0].dataset_shape[-1:] if len(inputs) == 1 else []
        levels, origin, notes = (
            (None, None, []) if needs is None
            else demand_levels(problem, needs)
        )
        demand = DemandWidth(
            columns=last[0] if last and isinstance(last[0], int)
            and last[0] > 0 else None,
            **described_names(
                dataset_section(path, "demand"), needs,
                ("units", "product", "date", "stock", "id")
            ),
            levels=levels, levels_from=origin, notes=notes,
            problem=None if needs is None else problem_name(problem)
        )
        outputs = None if levels is None else len(levels)

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
            table=table, anomaly=anomaly, uplift=uplift,
            price=price, demand=demand
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

    # 定義の要らない異常の検知の距離に基づく度合い (学習しない)
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
def list_problems(request: Request) -> ProblemList:
    """
    ML-Pathfinder の置き場にある問題の設定の一覧を返す関数

    Parameters
    ----------
    request: Request
        リクエスト (request.app は問題の設定の置き場を持つ)

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
    # problems/<識別子>/problem.yaml (置き場が無ければ空. ML-Fathom と同じ).
    # 保存先の problem_path と比べられるように, 置き場は実際のパスにする
    load_library("ml_quest", "handoff")
    root: Path = request.app.state.problems_dir
    files: list[Path] = []
    error = None
    try:
        root = root.resolve()
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


def contract_widths(kind: str, info: DatasetInputs) -> dict[str, int]:
    """
    データセットから入力と出力を作るときの数を, 骨組みの約束の入力の形の
    最後の大きさの鍵の数にする関数

    Parameters
    ----------
    kind: str
        骨組みの型 (ML-Quest の model_contract の skeleton)
    info: DatasetInputs
        データセットから読んだ入力と出力 (read_dataset_inputs)

    Returns
    ----------
    widths: dict[str, int]
        最後の大きさの鍵 (features, users, items) から数 (分からない鍵と,
        数を求めない骨組みは無し)
    """
    # 骨組みの型ごとに, 同じ意味の数だけを使う形 (別の受け持ちの数との
    # 比べを避けるため)
    widths = {}
    parts = {"anomaly": info.anomaly, "uplift": info.uplift,
             "price": info.price, "demand": info.demand}
    part = parts.get(kind)
    if kind == "table" and info.table is not None:
        widths["features"] = info.table.width
    elif kind == "rank" and info.rank is not None:
        widths["users"] = info.rank.users
        widths["items"] = info.rank.items
    elif part is not None and part.columns is not None:
        widths["features"] = part.columns

    return widths


def dataset_widths(
    target: ProjectTarget, kind: str
) -> tuple[dict[str, int], Optional[str]]:
    """
    保存先の記録のデータセットから, 骨組みの約束の最後の大きさの数を
    求める関数

    「データセットから入力と出力を作る」と同じ読み方 (同じ数で作った定義を
    約束に合うとみなすため)

    Parameters
    ----------
    target: ProjectTarget
        保存先 (記録のデータセットと問題の設定のファイル)
    kind: str
        骨組みの型

    Returns
    ----------
    widths: dict[str, int]
        最後の大きさの鍵から数 (読めなければ空)
    note: Optional[str]
        数を求められない理由 (求められれば None)
    """
    # 記録が無いか読めないデータセットは, 名前と出力だけの確かめ (数は
    # 確かめの助けなので, 読めない理由を問わず保存を止めない)
    note: Optional[str]
    if target.dataset is None:
        note = ("問題の設定にデータセットの場所の記録が無いので, データ"
                "セットで決まる形の最後の大きさは確かめていません")
        return {}, note
    try:
        info = read_dataset_inputs(DatasetBody(
            path=target.dataset, problem=target.problem_path
        ))
    except Exception as error:
        reason = (error.detail if isinstance(error, HTTPException)
                  else error)
        note = (f"記録のデータセット {target.dataset} を読めないので, データ"
                f"セットで決まる形の最後の大きさは確かめていません: {reason}")
        return {}, note
    widths = contract_widths(kind, info)
    note = None

    return widths, note


def contract_findings(
    target: ProjectTarget, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """
    定義が, 保存先の問題の設定の骨組みの学習のスクリプトが読む入力と出力の
    約束に合うかを確かめる関数

    約束は ML-Quest の model_contract と check_model_contract が持ち, この
    アプリには写さない. 定義の出力は形を持たないので, 出力の形の大きさ
    だけは, ML-ModelControl の形の推定の結果と約束の数をこのアプリで比べる

    Parameters
    ----------
    target: ProjectTarget
        保存先 (問題の設定から決めたものだけを確かめる)
    spec: dict[str, Any]
        保存する定義

    Returns
    ----------
    found: list[dict[str, Any]]
        合わない所 (level, id, message. id は直す場所)
    notes: list[str]
        確かめの知らせ (最後の大きさを確かめていない理由など)
    widths: dict[str, int]
        比べた大きさの数 (データセットの数と学ぶ水準の数 levels)

    Raises
    ----------
    HTTPException
        ML-Quest が無いか古い場合は 501, 問題の設定が無い場合は 404, 読め
        ない場合と確かめられない場合は 422
    """
    # 問題の設定の無い保存先と, 約束の無い骨組みは確かめ無し
    if target.problem_path is None:
        return [], [], {}
    quest = load_library("ml_quest", "check_model_contract")
    problem = read_problem(Path(target.problem_path))
    try:
        contract = quest.model_contract(problem)
    except READ_ERRORS as error:
        raise HTTPException(
            422, f"骨組みの入力と出力の約束を読めません: {error}"
        ) from error
    if contract is None:
        return [], [], {}

    # 最後の大きさは, データセットから入力と出力を作るときと同じ数. 学ぶ
    # 水準の数はデータセットに依らず, 学習の設定か問題の設定から
    kind = str(contract.get("skeleton"))
    widths, note = dataset_widths(target, kind)
    levels = learned_levels(problem) if kind == "demand" else None
    if levels is not None:
        widths["levels"] = levels
    try:
        found = quest.check_model_contract(problem, spec, widths)
    except READ_ERRORS as error:
        raise HTTPException(
            422, f"骨組みの入力と出力の約束と比べられません: {error}"
        ) from error
    notes = [] if note is None else [note]
    if note is None:
        notes += unchecked_widths(contract, widths)

    # 出力の形の大きさは, 定義の形の推定の結果と約束の比べ
    sized, unsized = output_size_findings(contract, spec, widths)
    found = [*found, *sized]
    notes += unsized

    return found, notes, widths


def learned_levels(problem: Any) -> Optional[int]:
    """
    時間をまたぐ量の判断の学ぶ分位点の水準の数を, データセットに依らず
    求める関数

    「データセットから入力と出力を作る」と同じ決め方 (記録のプロジェクトの
    学習の設定の demand.levels を先, 無ければ問題の設定の受け渡し)

    Parameters
    ----------
    problem: Any
        問題の設定 (read_problem)

    Returns
    ----------
    count: Optional[int]
        水準の数 (時間をまたぐ量の判断でないか, どちらも読めなければ None.
        その場合は約束の size で見る)
    """
    # 受け持ちの答えと, 学習の設定を先にした水準
    answers = demand_answers(problem)
    if answers is None:
        return None
    levels = demand_levels(problem, answers)[0]
    count = None if levels is None else len(levels)

    return count


def unchecked_widths(contract: Any, widths: dict[str, int]) -> list[str]:
    """
    データセットを読めたのに数を求められず, 確かめなかった入力の形の最後の
    大きさの知らせを作る関数

    Parameters
    ----------
    contract: Any
        骨組みの約束 (ML-Quest の model_contract)
    widths: dict[str, int]
        求めた最後の大きさの鍵から数

    Returns
    ----------
    notes: list[str]
        確かめなかった入力と出力ごとの知らせ (全て確かめたなら空)
    """
    # 約束の数も渡した数も無い, 鍵を持つ入力と出力 (崩れた約束は知らせ
    # 無し)
    notes = []
    for key, word in (("inputs", "入力"), ("outputs", "出力")):
        items = contract.get(key) if isinstance(contract, dict) else None
        notes += [
            f"{item.get('label') or item['width']} をデータセットから求め"
            f"られないので, {word} {item.get('name') or item.get('text')} "
            "の形の最後の大きさは確かめていません"
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and item.get("width")
            and item.get("size") is None and item["width"] not in widths
        ]

    return notes


def output_pairs(
    contract: Any, outputs: dict[str, Any]
) -> list[tuple[dict[str, Any], str]]:
    """
    大きさの決まる約束の出力の項目に, 定義の出力の名前を当てる関数

    Parameters
    ----------
    contract: Any
        骨組みの約束 (ML-Quest の model_contract)
    outputs: dict[str, Any]
        定義の出力 (名前から出どころ)

    Returns
    ----------
    pairs: list[tuple[dict[str, Any], str]]
        項目と定義の出力の名前の組 (定義に無い名前と, 数が合わず当てられ
        ない名前を問わない項目は除く. 名前と数の誤りは約束の比べが出す)
    """
    # 名前を決めた項目は同じ名前, 名前を問わない項目は他の出力に順に当て
    # (数が合う場合だけ)
    listed = contract.get("outputs") if isinstance(contract, dict) else None
    items = [item for item in (listed if isinstance(listed, list) else [])
             if isinstance(item, dict)]
    names = [item.get("name") for item in items if item.get("name")]
    pairs = [(item, str(item["name"])) for item in items
             if item.get("name") and item["name"] in outputs]
    free = [item for item in items if not item.get("name")]
    rest = [str(name) for name in outputs if name not in names]
    if free and len(free) == len(rest):
        pairs += list(zip(free, rest, strict=True))
    pairs = [(item, name) for item, name in pairs
             if item.get("width") or item.get("size") is not None]

    return pairs


def inferred_shapes(spec: dict[str, Any]) -> dict[str, Any]:
    """
    定義の形を推定する関数

    Parameters
    ----------
    spec: dict[str, Any]
        保存する定義

    Returns
    ----------
    result: dict[str, Any]
        ML-ModelControl の infer の結果 (推定で起きた誤りは, issues の
        1 件の誤りにした空の結果)
    """
    # 推定は定義を書き換えるので写しで (推定で起きた誤りは, 確かめの助け
    # なので保存を止めない)
    try:
        result: dict[str, Any] = infer(copy.deepcopy(spec))
    except Exception as error:
        result = {
            "issues": [{"severity": "error", "node": None,
                        "message": f"{type(error).__name__}: {error}"}],
            "nodes": {}, "outputs": {}, "estimated_only": False,
            "params": None
        }

    return result


def unsized_reason(result: dict[str, Any], ref: Any, name: str) -> str:
    """
    出力の形を推定できなかった理由を, 出力の出どころのノードの結果と推定
    の上限から選ぶ関数

    Parameters
    ----------
    result: dict[str, Any]
        形の推定の結果 (inferred_shapes)
    ref: Any
        出力の出どころ (定義の outputs の値. ノードの名前と出力の名前)
    name: str
        出力の名前

    Returns
    ----------
    reason: str
        理由の文
    """
    # 出力そのものの誤り, 出どころのノードの誤り, 推定の上限, 定義の誤りの
    # 順
    value = result["outputs"].get(name)
    node = result["nodes"].get(str(ref).partition(":")[0])
    issues = result["issues"]
    limits = [str(issue.get("message")) for issue in issues
              if issue.get("code") == "input_too_large"]
    errors = [str(issue.get("message")) for issue in issues
              if issue.get("severity") == "error"]
    if isinstance(value, dict) and value.get("error"):
        reason = str(value["error"])
    elif isinstance(node, dict) and node.get("error"):
        reason = str(node["error"])
    elif result.get("estimated_only"):
        reason = (f"パラメータ数 ({result.get('params')}) が形の推定の上限"
                  "を超えるので, 形を推定しません")
    elif limits:
        reason = limits[0]
    elif errors:
        reason = errors[0]
    else:
        reason = "形を推定できません"

    return reason


def shape_reason(value: Any) -> Optional[str]:
    """
    推定した出力の値が, バッチの次元を持つテンソルでない理由を返す関数

    Parameters
    ----------
    value: Any
        出力の形の推定の結果 (infer の outputs の値)

    Returns
    ----------
    reason: Optional[str]
        理由の文 (バッチの次元を持つテンソルなら None)
    """
    # テンソルでない値と, 次元の無いテンソル
    kind = value.get("kind") if isinstance(value, dict) else None
    shape = value.get("shape") if isinstance(value, dict) else None
    reason = None
    if kind != "tensor" or not isinstance(shape, list):
        reason = f"出力がテンソルではありません (種類 {kind})"
    elif not shape:
        reason = "出力がバッチの次元を持たないテンソルです (形 [])"

    return reason


def batch_free_size(shape: list[int]) -> int:
    """
    バッチの次元を除いた大きさ (残りの次元の大きさの積) を求める関数

    出力を (件数,) か (件数, 数) に reshape して読む骨組み
    (RESHAPED_SKELETONS) では, 最後の大きさでなく積で比べる ((件数,) は 1)

    Parameters
    ----------
    shape: list[int]
        出力の形 (最初がバッチの次元)

    Returns
    ----------
    size: int
        バッチの次元を除いた大きさ
    """
    # 残りの次元の積
    size = 1
    for value in shape[1:]:
        size *= value

    return size


def output_size_findings(
    contract: Any, spec: dict[str, Any], widths: dict[str, int]
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    定義の出力の形の大きさを, 形の推定で求めて約束の数と比べる関数

    定義の出力は形を持たないので, 定義の入力の形から ML-ModelControl の
    形の推定で求める (入力の形の誤りは入力の比べが出すので, 入力の形は
    定義のまま)

    Parameters
    ----------
    contract: Any
        骨組みの約束 (ML-Quest の model_contract)
    spec: dict[str, Any]
        保存する定義
    widths: dict[str, int]
        データセットから求めた最後の大きさの鍵から数

    Returns
    ----------
    found: list[dict[str, Any]]
        合わない所 (level, id, message. id は outputs.<名前>)
    notes: list[str]
        形を推定できず確かめなかった出力ごとの知らせ
    """
    # 渡した数を約束の数より先に (入力の比べと同じ). 数の無い出力は確かめ
    # 無し (知らせは unchecked_widths)
    listed = spec.get("outputs")
    outputs = listed if isinstance(listed, dict) else {}
    pairs: list[tuple[dict[str, Any], str, int]] = []
    for item, name in output_pairs(contract, outputs):
        size = widths.get(str(item.get("width")), item.get("size"))
        if isinstance(size, int):
            pairs.append((item, name, size))
    if not pairs:
        return [], []

    # 推定した形の大きさと約束の数の比べ (reshape して読む骨組みはバッチの
    # 次元を除いた積)
    reshaped = contract.get("skeleton") in RESHAPED_SKELETONS
    result = inferred_shapes(spec)
    found = []
    notes = []
    for item, name, size in pairs:
        value = result["outputs"].get(name)
        if value is None or (isinstance(value, dict) and "error" in value):
            why = unsized_reason(result, outputs[name], name)
            notes.append(f"出力 {name} の形を推定できないので, 大きさは確か"
                         f"めていません: {why}")
            continue
        why_not = shape_reason(value)
        if why_not is not None:
            notes.append(f"出力 {name} の大きさは確かめていません: "
                         f"{why_not}")
            continue
        found += size_findings(item, name, size, value["shape"], reshaped)

    return found, notes


def size_findings(
    item: dict[str, Any], name: str, size: int, shape: list[int],
    reshaped: bool
) -> list[dict[str, Any]]:
    """
    出力の形の大きさを, 約束の数と比べる関数

    Parameters
    ----------
    item: dict[str, Any]
        約束の出力の項目
    name: str
        定義の出力の名前
    size: int
        骨組みが読む数
    shape: list[int]
        推定した出力の形 (最初がバッチの次元)
    reshaped: bool
        骨組みが出力を reshape して読むか (真ならバッチの次元を除いた積,
        偽なら (件数, 数) の 2 次元で比べる)

    Returns
    ----------
    found: list[dict[str, Any]]
        合わない所 (合えば空. 2 次元は最後の大きさ, 他は形を添えた文)
    """
    # reshape して読む骨組みは積, 他は (件数, 数) の形
    label = item.get("label")
    expected = f"{label} {size}" if label else f" {size}"
    actual: Optional[int]
    if reshaped:
        actual = batch_free_size(shape)
        message = (
            f"出力 {name} のバッチの次元を除いた大きさ (残りの次元の積) "
            f"を{expected} にしてください (今は形 {shape} で {actual})"
        )
    else:
        actual = shape[-1] if len(shape) == 2 else None
        message = (
            f"出力 {name} を (件数, {expected.strip()}) の 2 次元の形に"
            "してください (骨組みは出力をそのまま使います. 今は形 "
            f"{shape})"
        )
    if actual == size:
        return []
    if len(shape) == 2:
        message = (f"出力 {name} の最後の大きさを{expected} にしてください "
                   f"(今は {shape[-1]})")
    found = [{"level": "error", "id": f"outputs.{name}", "message": message}]

    return found


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
        書いたファイル, 更新時刻と, 骨組みの約束の確かめの知らせ

    Raises
    ----------
    HTTPException
        確かめた保存先 (target) と違う場合と, 既にあり上書きしない場合は
        409, 定義を読めない場合は 422, 骨組みの約束に合わない所があり
        accept_contract でない場合は 409 (detail に画面の文 message, 直す
        場所 findings, 知らせ notes と比べた数 widths), 書けない場合は 500
        (他は保存先を決めるときと同じ)
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

    # 問題の設定の骨組みの約束に合わない所があれば, 書くかは利用者の選択
    # (学習を始めてから食い違いに気付かないため. 直さずに書く指定では確かめ
    # 無し)
    found, notes, widths = (
        ([], [], {}) if body.accept_contract
        else contract_findings(target, body.spec)
    )
    if found:
        raise HTTPException(409, {
            "message": (
                f"骨組みの学習のスクリプトが読む入力と出力に合わない所が "
                f"{len(found)} 件あります. 直してから保存するか, 直さずに"
                "保存するかを選んでください"
            ),
            "findings": found, "notes": notes, "widths": widths,
        })

    # 保存先までの無いディレクトリは, プロジェクトの中だけ作る
    try:
        make_parents(path, target.project)
        updated_at = write_atomic(path, text)
    except OSError as error:
        raise HTTPException(
            500, f"{path} に書けません: {error}"
        ) from error
    saved = ProjectSaved(path=str(path), updated_at=updated_at, notes=notes)

    return saved
