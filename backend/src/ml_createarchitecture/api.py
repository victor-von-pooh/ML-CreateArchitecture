from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from ml_modelcontrol import (
    SpecError,
    describe_units,
    dump_spec,
    export_python,
    infer,
    list_examples,
    load_example,
    load_spec,
    validate,
)
from pydantic import BaseModel

from ml_createarchitecture.store import (
    ModelError,
    ModelNotFoundError,
    ModelStore,
)

router = APIRouter(prefix="/api")


class Issue(BaseModel):
    # 誤りか注意か, 識別子, 説明, ノードの名前, 場所
    severity: str
    code: str
    message: str
    node: Optional[str] = None
    location: Optional[str] = None


class NodeResult(BaseModel):
    # 計算の結果 (ok, error, skipped), 出力の口ごとの形, パラメータ数, エラー
    status: str
    outputs: dict[str, Any] = {}
    params: int = 0
    trainable: int = 0
    error: Optional[str] = None


class CheckResult(BaseModel):
    # 誤りと注意, 形の推定をしたか, ノードごとの結果, 出力の形, パラメータ数,
    # 上限を超えて計算しなかったか, 保存する形の YAML
    issues: list[Issue]
    inferred: bool
    ok: bool = False
    nodes: dict[str, NodeResult] = {}
    outputs: dict[str, Any] = {}
    params: Optional[int] = None
    trainable: Optional[int] = None
    estimated_only: bool = False
    yaml: str = ""


class SpecBody(BaseModel):
    # モデルの定義と, 形の推定もするか
    spec: dict[str, Any]
    infer: bool = True


class ExportBody(BaseModel):
    # モデルの定義と, 書き出すクラスの名前
    spec: dict[str, Any]
    class_name: str = "Model"


class ExportResult(BaseModel):
    # 書き出した Python のコード
    code: str


class ModelSummary(BaseModel):
    # 定義の名前と更新時刻
    name: str
    updated_at: str


class ModelFile(BaseModel):
    # 定義の名前と中身
    name: str
    spec: dict[str, Any]


def get_store(request: Request) -> ModelStore:
    """
    アプリに保存した定義のファイルの置き場を取り出す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    store: ModelStore
        定義のファイルの置き場
    """
    store: ModelStore = request.app.state.store

    return store


# 形の推定で作る入力の例の要素数の上限 (大きすぎる形でメモリを使い切らない)
MAX_INPUT_ELEMENTS = 10_000_000


def input_too_large(spec: dict[str, Any]) -> Optional[str]:
    """
    入力の例の要素数が上限を超えるかを確かめる関数

    Parameters
    ----------
    spec: dict[str, Any]
        モデルの定義

    Returns
    ----------
    message: Optional[str]
        超える場合の説明, 超えない場合は None
    """
    # 入力ごとに形の要素数を掛け合わせて確認
    message = None
    inputs = spec.get("inputs")
    if not isinstance(inputs, dict):
        return message
    for name, value in inputs.items():
        shape = value.get("shape") if isinstance(value, dict) else None
        if not isinstance(shape, list):
            continue
        count = 1
        for size in shape:
            if isinstance(size, int) and size > 0:
                count *= size
        if count > MAX_INPUT_ELEMENTS:
            message = (
                f"入力 {name} の例の要素数 ({count}) が上限 "
                f"({MAX_INPUT_ELEMENTS}) を超えるので, 形を推定しません"
            )

    return message


@router.get("/units")
def list_units() -> list[dict[str, Any]]:
    """
    部品の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    units: list[dict[str, Any]]
        ML-ModelControl の部品ごとの名前, 説明, 引数, 口
    """
    units: list[dict[str, Any]] = describe_units()

    return units


@router.post("/check")
def check_spec(body: SpecBody) -> CheckResult:
    """
    モデルの定義を確かめ, 誤りがなければ形を推定する関数

    Parameters
    ----------
    body: SpecBody
        モデルの定義と, 形の推定もするか

    Returns
    ----------
    result: CheckResult
        誤りと注意, ノードごとの形とパラメータ数, YAML
    """
    # 確かめて, 誤りが無ければ形を推定 (誤りがある場合は確認の結果だけ)
    # 予期しないエラー (極端な形など) は, 定義全体の誤りとして返却
    try:
        result = run_check(body)
    except Exception as error:
        result = CheckResult(
            issues=[Issue(
                severity="error", code="check_failed",
                message=f"確認できません: {type(error).__name__}: {error}"
            )],
            inferred=False
        )

    return result


def run_check(body: SpecBody) -> CheckResult:
    """
    確認と形の推定を行う関数

    Parameters
    ----------
    body: SpecBody
        モデルの定義と, 形の推定もするか

    Returns
    ----------
    result: CheckResult
        誤りと注意, ノードごとの形とパラメータ数, YAML
    """
    # 定義として読めない場合と, 入力の例の要素数が大きすぎる場合は形を
    # 推定しない. 他の誤りがあっても, 誤りのあるノードとその先を除いて
    # 推定する (組み立ての途中でも形を確かめられるように)
    issues = validate(body.spec)
    unreadable = any(issue["code"] == "invalid_spec" for issue in issues)
    yaml_text = ""
    if not unreadable:
        yaml_text = dump_spec(load_spec(body.spec))
    too_large = input_too_large(body.spec)
    if too_large is not None:
        issues.append({
            "severity": "warning", "code": "input_too_large",
            "message": too_large, "node": None, "location": "inputs"
        })
    if unreadable or not body.infer or too_large is not None:
        result = CheckResult(
            issues=[Issue(**issue) for issue in issues], inferred=False,
            yaml=yaml_text
        )
        return result
    inferred = infer(body.spec)
    result = CheckResult(
        issues=[Issue(**issue) for issue in inferred["issues"]],
        inferred=bool(inferred["nodes"]), ok=inferred["ok"],
        nodes={
            name: NodeResult(**value)
            for name, value in inferred["nodes"].items()
        },
        outputs=inferred["outputs"], params=inferred["params"],
        trainable=inferred["trainable"],
        estimated_only=inferred["estimated_only"], yaml=yaml_text
    )

    return result


@router.post("/export")
def export_spec(body: ExportBody) -> ExportResult:
    """
    モデルの定義から Python のコードを書き出す関数

    Parameters
    ----------
    body: ExportBody
        モデルの定義とクラスの名前

    Returns
    ----------
    result: ExportResult
        Python のコード
    """
    # 定義に誤りがある場合と, 書き出せない場合は 400
    try:
        code = export_python(body.spec, class_name=body.class_name)
    except (SpecError, ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    result = ExportResult(code=code)

    return result


class ExampleSummary(BaseModel):
    # 例の名前と説明
    name: str
    description: str


@router.get("/examples")
def list_example_models() -> list[ExampleSummary]:
    """
    ML-ModelControl に同梱した例のモデルの一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    examples: list[ExampleSummary]
        例の名前と説明
    """
    examples = [ExampleSummary(**item) for item in list_examples()]

    return examples


@router.get("/examples/{name}")
def get_example(name: str) -> ModelFile:
    """
    例のモデルの定義を読み込む関数

    Parameters
    ----------
    name: str
        例の名前

    Returns
    ----------
    model: ModelFile
        例の名前と定義
    """
    # 一覧に無い名前は 404
    try:
        spec = load_example(name)
    except KeyError as error:
        raise HTTPException(404, f"例 {name} はありません") from error
    model = ModelFile(name=name, spec=spec)

    return model


@router.get("/models")
def list_models(request: Request) -> list[ModelSummary]:
    """
    保存した定義の一覧を返す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    models: list[ModelSummary]
        名前と更新時刻 (新しい順)
    """
    models = [
        ModelSummary(**item) for item in get_store(request).list_models()
    ]

    return models


@router.get("/models/{name}")
def get_model(name: str, request: Request) -> ModelFile:
    """
    保存した定義を読み込む関数

    Parameters
    ----------
    name: str
        定義の名前
    request: Request
        リクエスト

    Returns
    ----------
    model: ModelFile
        定義の名前と中身
    """
    # 無いか読めない場合は 404
    try:
        spec = get_store(request).load(name)
    except ModelError as error:
        raise HTTPException(404, str(error)) from error
    model = ModelFile(name=name, spec=spec)

    return model


@router.put("/models/{name}")
def save_model(name: str, body: SpecBody, request: Request) -> ModelSummary:
    """
    定義を保存する関数 (誤りがあっても途中の状態として保存できる)

    Parameters
    ----------
    name: str
        定義の名前
    body: SpecBody
        モデルの定義
    request: Request
        リクエスト

    Returns
    ----------
    summary: ModelSummary
        保存した定義の名前と更新時刻
    """
    # 名前の規則に合わない場合と, 版の読めない定義は 400
    try:
        saved = get_store(request).save(name, body.spec)
    except ModelError as error:
        raise HTTPException(400, str(error)) from error
    summary = ModelSummary(**saved)

    return summary


@router.delete("/models/{name}", status_code=204)
def delete_model(name: str, request: Request) -> None:
    """
    保存した定義を消す関数

    Parameters
    ----------
    name: str
        定義の名前
    request: Request
        リクエスト

    Returns
    ----------
    None
    """
    # 名前の規則に合わない場合は 400, 無い場合は 404, 消せない場合は 500
    store = get_store(request)
    try:
        store.path_for(name)
    except ModelError as error:
        raise HTTPException(400, str(error)) from error
    try:
        store.delete(name)
    except ModelNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except ModelError as error:
        raise HTTPException(500, str(error)) from error
