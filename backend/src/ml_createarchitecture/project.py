"""
データセットとプロジェクトにつなぐ API

データセットの説明は ML-EDA, 問題の設定は ML-Quest を通して読む (他の段階
のファイルは, その約束を持つライブラリを通して読む決まり). どちらも追加の
依存 project で入る
"""

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from ml_createarchitecture.store import ModelError, spec_text, write_atomic

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

# プロジェクトのディレクトリの問題の設定
PROBLEM_FILE = "problem.yaml"

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


class DatasetInput(BaseModel):
    # モデルの入力の名前, 形の例 (バッチの次元を含む), 型
    name: str
    shape: list[int]
    dtype: str


class DatasetInputs(BaseModel):
    # データセットの場所と, ML-EDA の model_inputs の結果
    path: str
    inputs: list[DatasetInput]
    task: Optional[str] = None
    target: Optional[str] = None
    classes: Optional[list[Any]] = None
    outputs: Optional[int] = None


class ProjectTarget(BaseModel):
    # モデルの定義の保存先と, どこから決めたか
    path: str
    source: str = Field(description="problem (問題の設定の記録) か project")
    exists: bool
    project: Optional[str] = None
    dataset: Optional[str] = None
    problem: Optional[str] = Field(
        default=None, description="問題の設定の名前"
    )


class ProjectSaveBody(BaseModel):
    # 保存先を決める場所, 定義, 既にある場合に上書きするか
    path: str
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
        入力 (形はバッチの次元を例の大きさにしたもの), タスク, 目的の列,
        クラス, 出す値の数

    Raises
    ----------
    HTTPException
        ML-EDA が無い場合は 501, ディレクトリか説明が無い場合は 404,
        読めない場合 (版 2, 知らない版など) は 422
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
            dtype=str(item.get("dtype") or "float32")
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


def resolve_target(text: str) -> ProjectTarget:
    """
    プロジェクトのディレクトリか問題の設定のファイルから, モデルの定義の
    保存先を決める関数

    Parameters
    ----------
    text: str
        プロジェクトのディレクトリか問題の設定のファイル

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
                422,
                f"{path.name} にプロジェクトの記録 (model.project_model_spec)"
                " がありません. ML-Pathfinder でプロジェクトを書き出してから"
                "選んでください (書き出した後なら, ML-Quest が古いので "
                f"{SETUP_HINT} で入れ直す)"
            )
        target = check_target(found)
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
        # リンクをたどった先もプロジェクトの中に限る
        spec_path = (path / DEFAULT_SPEC).resolve()
        if not spec_path.is_relative_to(path):
            raise HTTPException(
                422, f"保存先 {spec_path} がプロジェクト {path} の外にあります"
            )
        found = ProjectTarget(
            path=str(spec_path), source="project", exists=spec_path.exists(),
            project=str(path), dataset=config_dataset(path)
        )
    target = check_target(found)

    return target


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
        拡張子が YAML でない場合と, ディレクトリなどファイルでないものが
        ある場合は 422
    """
    # YAML のファイルだけに書く
    path = Path(target.path)
    if path.suffix.lower() not in SPEC_SUFFIXES:
        raise HTTPException(
            422, f"保存先 {path} は YAML のファイル (.yaml) ではありません"
        )
    if path.exists() and not path.is_file():
        raise HTTPException(
            422, f"保存先 {path} はファイルではありません"
        )

    return target


@router.post("/project/target")
def project_target(body: PathBody) -> ProjectTarget:
    """
    プロジェクトのモデルの定義の保存先を返す関数

    Parameters
    ----------
    body: PathBody
        プロジェクトのディレクトリか問題の設定のファイル

    Returns
    ----------
    target: ProjectTarget
        保存先, プロジェクトとデータセットの場所, 既にあるか
    """
    target = resolve_target(body.path)

    return target


@router.put("/project/model")
def save_project_model(body: ProjectSaveBody) -> ProjectSaved:
    """
    定義をプロジェクトの保存先に書く関数

    Parameters
    ----------
    body: ProjectSaveBody
        保存先を決める場所, 定義, 上書きするか

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
    target = resolve_target(body.path)
    path = Path(target.path)
    if path.exists() and not body.overwrite:
        raise HTTPException(409, f"{path} が既にあります")
    try:
        text = spec_text(body.spec)
    except ModelError as error:
        raise HTTPException(422, str(error)) from error

    # 保存先のディレクトリ (骨組みの models/) が無ければ作ってから書き込み
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        updated_at = write_atomic(path, text)
    except OSError as error:
        raise HTTPException(
            500, f"{path} に書けません: {error}"
        ) from error
    saved = ProjectSaved(path=str(path), updated_at=updated_at)

    return saved
