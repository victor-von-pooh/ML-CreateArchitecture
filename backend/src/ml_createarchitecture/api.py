import threading
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from ml_modelcontrol import (
    SpecError,
    build_model,
    describe_units,
    dump_spec,
    export_python,
    get_unit,
    infer,
    list_examples,
    load_example,
    load_spec,
    pretrained_info,
    validate,
)
from ml_modelcontrol.pretrained import gated_error
from pydantic import BaseModel

from ml_createarchitecture.settings import MODULES_ENV
from ml_createarchitecture.store import (
    ModelError,
    ModelNotFoundError,
    ModelStore,
)

router = APIRouter()


class Issue(BaseModel):
    # 誤りか注意か, 識別子, 説明, ノードの名前, 場所, 部分モデルの名前
    severity: str
    code: str
    message: str
    node: Optional[str] = None
    location: Optional[str] = None
    # 部分モデルの中の誤りの場合の部分モデルの名前
    subgraph: Optional[str] = None


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


class CustomModule(BaseModel):
    # 起動の設定に書いたモジュールの名前と, 読み込んだ部品の名前
    name: str
    units: list[str]


class ModulesInfo(BaseModel):
    # 起動の設定の環境変数の名前と, 読み込んだモジュール (書いた順)
    env: str
    modules: list[CustomModule]


@router.get("/modules")
def list_custom_modules(request: Request) -> ModulesInfo:
    """
    起動の設定で読み込んだ自作の部品のモジュールを返す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    info: ModulesInfo
        環境変数の名前と, モジュールごとの部品の名前
    """
    loaded: dict[str, list[str]] = request.app.state.custom_modules
    info = ModulesInfo(
        env=MODULES_ENV,
        modules=[
            CustomModule(name=name, units=units)
            for name, units in loaded.items()
        ]
    )

    return info


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


class PretrainedBody(BaseModel):
    # 部品の名前と引数の値
    unit: str
    params: dict[str, Any] = {}


class PretrainedInfo(BaseModel):
    # 学習済みの重みのライセンスの情報 (ML-ModelControl の pretrained_info).
    # 分類 (commercial, noncommercial, unknown), 警告するか, 画面に出す説明,
    # どこから読んだか, 推論の一式に写せるか, 読んだ版, 作者, 表示の義務,
    # 出どころの表記の例, 重みの大きさ (バイト数)
    unit: str
    library: str
    model: str
    hub_id: Optional[str] = None
    revision: Optional[str] = None
    pinned_revision: Optional[str] = None
    license: Optional[str] = None
    licenses: list[str] = []
    confirmed: bool = False
    status: str
    redistributable: bool
    gated: bool
    alert: bool
    message: str
    source: str
    url: Optional[str] = None
    author: Optional[str] = None
    attribution: str = ""
    credit: Optional[str] = None
    size_bytes: Optional[int] = None


@router.post("/pretrained")
def read_pretrained(body: PretrainedBody) -> Optional[PretrainedInfo]:
    """
    ノードが読む学習済みの重みのライセンスを, 配布元から読んで返す関数

    結果は ML-ModelControl がプロセスの中に覚えるので, 同じモデルを何度
    選んでもネットワークは 1 回だけ使う

    Parameters
    ----------
    body: PretrainedBody
        部品の名前と引数の値

    Returns
    ----------
    info: Optional[PretrainedInfo]
        ライセンスの情報, 重みを読まない場合は null
    """
    # 登録簿に無い部品は 404 (情報を読む途中の誤りと分けるため先に確認)
    try:
        get_unit(body.unit)
    except KeyError as error:
        raise HTTPException(
            404, f"部品 {body.unit} は登録されていません"
        ) from error
    found = pretrained_info(body.unit, body.params)
    info = None if found is None else PretrainedInfo(**found)

    return info


class WeightsBody(BaseModel):
    # モデルの定義
    spec: dict[str, Any]


class WeightsLicense(PretrainedInfo):
    # 外部の重みのライセンス (Hugging Face の置き場から読む場合は部品が無い)
    unit: Optional[str] = None  # type: ignore[assignment]


class WeightsReport(BaseModel):
    # 外部の重みを読んだノード (部分モデルの中は . でつなぐ), 出どころ,
    # 合わない重みで止めるか, 読めた / 形の違う / 新しい / 使わない重みの
    # 数とキーの一部 (形の違うものはキー, ファイルの形, ノードの形),
    # ライセンス
    node: str
    source: str
    strict: bool
    loaded: int
    mismatched: int
    new: int
    unused: int
    mismatched_keys: list[list[Any]] = []
    new_keys: list[str] = []
    unused_keys: list[str] = []
    license: Optional[WeightsLicense] = None
    # 注意の文 (共有するパラメータをファイルから読まなかったなど)
    warnings: list[str] = []


class GatedInfo(BaseModel):
    # 申請制の重みを読めなかった理由 (no_token, invalid_token,
    # not_granted), 置き場の名前, 配布元のページ
    reason: str
    hub_id: Optional[str] = None
    url: Optional[str] = None


class WeightsResult(BaseModel):
    # 読めたか, ノードごとの報告, 読めない場合の誤りの文, 申請制の重みを
    # 読めなかった場合の理由
    ok: bool
    reports: list[WeightsReport] = []
    error: Optional[str] = None
    gated: Optional[GatedInfo] = None


# 外部の重みの読み込みを 1 つずつにする鍵 (大きな重みを並べて読まないため)
WEIGHTS_LOCK = threading.Lock()


def uses_weights(spec: dict[str, Any]) -> bool:
    """
    定義 (本体か部分モデル) のノードに外部の重み (weights) があるかを返す
    関数

    Parameters
    ----------
    spec: dict[str, Any]
        モデルの定義

    Returns
    ----------
    found: bool
        weights を持つノードがあれば True
    """
    # 本体と部分モデルのノードの確認
    graphs = [spec]
    subgraphs = spec.get("subgraphs")
    if isinstance(subgraphs, dict):
        graphs += list(subgraphs.values())
    nodes: list[Any] = []
    for graph in graphs:
        found_nodes = graph.get("nodes") if isinstance(graph, dict) else None
        if isinstance(found_nodes, dict):
            nodes += list(found_nodes.values())
    found = any(
        isinstance(node, dict) and "weights" in node for node in nodes
    )

    return found


def too_large_to_build(spec: dict[str, Any]) -> Optional[str]:
    """
    確認 (POST /api/check) と同じ見積もりで, 組み立てるには大きすぎるかを
    確かめる関数

    Parameters
    ----------
    spec: dict[str, Any]
        モデルの定義

    Returns
    ----------
    message: Optional[str]
        大きすぎる場合の説明, 組み立ててよい場合は None
    """
    # 入力の例の要素数と, パラメータ数の上限 (形の推定と同じ) の確認.
    # 読めない定義は組み立てで誤りの文になるので, ここでは止めない
    message = input_too_large(spec)
    if message is not None:
        return message
    try:
        estimated = infer(spec)
    except Exception:
        return message
    if estimated["estimated_only"]:
        message = (
            f"パラメータ数 ({estimated['params']}) が形の推定の上限を超える"
            "ので, 重みを読んで組み立てません"
        )

    return message


@router.post("/weights")
def read_weights(body: WeightsBody, request: Request) -> WeightsResult:
    """
    定義を組み立てて外部の重みを読み, ノードごとの報告を返す関数

    ファイルとネットワークを使い重みも作るので, 編集のたびの確認では
    呼ばず, 画面のボタンを押したときだけ呼ぶ

    Parameters
    ----------
    body: WeightsBody
        モデルの定義
    request: Request
        リクエスト (相対のパスの元にする定義の置き場を取り出す)

    Returns
    ----------
    result: WeightsResult
        読めたか, ノードごとの報告, 読めない場合の誤りの文
    """
    # 外部の重みが無ければ組み立てない (学習済みの重みの取得を避けるため).
    # 読んでいる途中なら待たずに知らせる
    if not uses_weights(body.spec):
        result = WeightsResult(ok=True)
        return result
    # 確認と同じ大きさの見積もりで, 大きすぎるモデルは組み立てない
    too_large = too_large_to_build(body.spec)
    if too_large is not None:
        result = WeightsResult(ok=False, error=too_large)
        return result
    if not WEIGHTS_LOCK.acquire(blocking=False):
        result = WeightsResult(
            ok=False, error="他の重みの読み込みの途中です. 終わってから押して"
            "ください"
        )
        return result

    # 相対のパスは定義の置き場から (保存した定義の置き場と同じ)
    # 定義の誤りと読み込みの誤りは, 誤りの文として返却
    base_dir = get_store(request).root
    try:
        model = build_model(body.spec, base_dir=base_dir)
    except SpecError as error:
        result = WeightsResult(ok=False, error=str(error))
        return result
    except Exception as error:
        result = build_error(error)
        return result
    finally:
        WEIGHTS_LOCK.release()
    result = WeightsResult(
        ok=True,
        reports=[WeightsReport(**item) for item in model.weights_report]
    )

    return result


def build_error(error: Exception) -> WeightsResult:
    """
    組み立てで起きた誤りを, 画面に出す誤りの文にする関数

    Parameters
    ----------
    error: Exception
        組み立てか重みの読み込みで起きた誤り

    Returns
    ----------
    result: WeightsResult
        読めなかった結果. 申請制の重みの誤りは, 型の名前を付けない日本語の
        理由と次にすること, 理由と配布元のページ. それ以外は型の名前を
        付けた誤りの文
    """
    # ノードの誤り (NodeBuildError) は申請制の誤りを持ち, 文にノードの
    # 名前がある. 他の道 (外部の重みの hub など) は原因の誤りから探す
    gated = getattr(error, "gated", None)
    message = str(error)
    if gated is None:
        gated = gated_error(error)
        message = str(gated)
    if gated is None:
        result = WeightsResult(
            ok=False, error=f"{type(error).__name__}: {error}"
        )
        return result
    result = WeightsResult(
        ok=False, error=message,
        gated=GatedInfo(
            reason=gated.reason, hub_id=gated.hub_id, url=gated.url
        )
    )

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
