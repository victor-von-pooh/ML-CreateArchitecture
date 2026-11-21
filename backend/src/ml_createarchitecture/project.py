"""
データセットとプロジェクトにつなぐ API

データセットの説明は ML-EDA, 問題の設定は ML-Quest を通して読む (他の段階
のファイルは, その約束を持つライブラリを通して読む決まり). どちらも追加の
依存 project で入る
"""

import datetime as dt
import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException
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


@router.post("/dataset/inputs")
def read_dataset_inputs(body: PathBody) -> DatasetInputs:
    """
    データセットから, モデルの入力と出す値の数を読む関数

    Parameters
    ----------
    body: PathBody
        データセットのディレクトリ

    Returns
    ----------
    result: DatasetInputs
        入力 (形はバッチの次元を例の大きさにしたものと, データセットの
        形), タスク, 目的の列, クラス, 正解の種類, 出す値の数

    Raises
    ----------
    HTTPException
        ML-EDA が無い場合は 501, ディレクトリか説明が無い場合は 404,
        読めない場合 (知らない版, 知らない項目の組など) は 422
    """
    # 説明の版と読み方は ML-EDA に任せる
    path = absolute_path(body.path)
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
    try:
        result = DatasetInputs(
            path=str(path), inputs=inputs, task=optional_text(info, "task"),
            target=optional_text(info, "target"),
            classes=list(classes) if isinstance(classes, list) else None,
            target_kind=optional_text(info, "target_kind"),
            outputs=outputs if isinstance(outputs, int)
            and not isinstance(outputs, bool) and outputs > 0 else None
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
        置き場の記録が無いか
    """
    # プロジェクトの場所の記録の有無で分ける (読み直しの間に読めなくなった
    # 場合は, 記録の無い文に)
    quest = load_library("ml_quest", "handoff")
    try:
        stages = quest.handoff(quest.load_problem(problem_path))
    except READ_ERRORS:
        stages = {}
    project_path = section(stages, "project").get("path")
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
        problem=name if isinstance(name, str) else None
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
    # 無いファイルは場所の文字で比べる
    same = first == second
    if not same and first.exists() and second.exists():
        try:
            same = os.path.samefile(first, second)
        except OSError:
            same = False

    return same


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
        拡張子が YAML でない場合, 学習の設定のディレクトリ (configs/) の中
        の場合と, ディレクトリなどファイルでないものがある場合は 422
    """
    # YAML のファイルだけに書く
    path = Path(target.path)
    if path.suffix.lower() not in SPEC_SUFFIXES:
        raise HTTPException(
            422, f"保存先 {path} は YAML のファイル (.yaml) ではありません"
        )

    # 学習の設定 (configs/ の下) は書き換えない. 大文字と小文字を区別しない
    # ファイルシステムでも同じ場所を断るため, 名前は大小を無視して比べる
    if target.project is not None and path.is_relative_to(target.project):
        parts = path.relative_to(target.project).parts
        if parts and parts[0].casefold() == CONFIG_DIR:
            raise HTTPException(
                422,
                f"保存先 {path} は学習の設定のディレクトリ ({CONFIG_DIR}/) "
                "の中です. モデルの定義は models/ などに置いてください"
            )
    if path.exists() and not path.is_file():
        raise HTTPException(
            422, f"保存先 {path} はファイルではありません"
        )

    return target


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
        既にあり上書きしない場合は 409, 定義を読めない場合と書けない場合は
        422 (他は保存先を決めるときと同じ)
    """
    # 押した時点の保存先を決め直し, 既にあれば上書きの確認を求める
    target = resolve_target(body.path, body.name)
    path = Path(target.path)
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
