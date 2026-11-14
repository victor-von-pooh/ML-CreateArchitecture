from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from ml_modelcontrol import (
    SpecError,
    describe_units,
    dump_spec,
    export_python,
    infer,
    load_spec,
    validate,
)
from pydantic import BaseModel

from ml_createarchitecture.store import ModelError, ModelStore

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
    issues = validate(body.spec)
    has_error = any(issue["severity"] == "error" for issue in issues)
    yaml_text = ""
    if not any(issue["code"] == "invalid_spec" for issue in issues):
        yaml_text = dump_spec(load_spec(body.spec))
    if has_error or not body.infer:
        result = CheckResult(
            issues=[Issue(**issue) for issue in issues], inferred=False,
            yaml=yaml_text
        )
        return result
    inferred = infer(body.spec)
    result = CheckResult(
        issues=[Issue(**issue) for issue in inferred["issues"]],
        inferred=not inferred["estimated_only"],
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
    # 定義に誤りがある場合は 400
    try:
        code = export_python(body.spec, class_name=body.class_name)
    except (SpecError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    result = ExportResult(code=code)

    return result


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
    store = get_store(request)
    try:
        store.save(name, body.spec)
    except ModelError as error:
        raise HTTPException(400, str(error)) from error
    summary = next(
        ModelSummary(**item) for item in store.list_models()
        if item["name"] == name
    )

    return summary
