"""
雛形のままの定義を見分け, データに合わせて書き換える API

「データセットから入力と出力を作る」で, 本体の定義が ML-ModelControl の
書き換えられる雛形 (TEMPLATE_ARGS にある同梱の例) のままなら, 画面がノード
をたどって引数を決めずに, 雛形の引数の値を adapt_example に渡して定義を
作る. 雛形ごとの書き換える場所を ML-ModelControl の 1 か所にまとめるため
"""

import copy
from typing import Any, Optional, Union

from fastapi import APIRouter, HTTPException
from ml_modelcontrol.templates import (
    TEMPLATE_ARGS,
    adapt_example,
    read_place,
    template_places,
)
from pydantic import BaseModel

router = APIRouter(prefix="/api")

# 照らさない定義の項目 (位置, 名前と説明. 画面と保存の名前で変わるため)
LOOSE_KEYS = ("ui", "name", "description")

# 照らさない入力の項目 (形と型はデータセットから入れ直すため)
LOOSE_INPUT_KEYS = ("ui", "shape", "dtype")

# 定義の値を読めない場合の誤り
READ_ERRORS = (KeyError, IndexError, TypeError, ValueError, AttributeError)


class TemplatePlace(BaseModel):
    # 種類 (input は入力の shape の軸, param はノードの引数, shape はノード
    # の引数 shape の要素), 入力かノードの名前, 軸か引数の名前, 倍率
    kind: str
    owner: str
    key: Union[int, str]
    scale: int


class TemplateArg(BaseModel):
    # 引数の名前, 今の値, 書き換える場所
    name: str
    value: int
    places: list[TemplatePlace]


class TemplateMatch(BaseModel):
    # 雛形の名前と, 書き換える引数ごとの今の値と場所
    name: str
    args: list[TemplateArg]


class TemplateChange(BaseModel):
    # 引数の名前, 前と後の値, 書き換えた場所
    name: str
    before: int
    after: int
    places: list[TemplatePlace]


class TemplateAdapted(BaseModel):
    # 雛形の名前, 書き換えた定義, 渡した引数ごとの前と後の値
    name: str
    spec: dict[str, Any]
    args: list[TemplateChange]


class MatchBody(BaseModel):
    # 照らす定義
    spec: dict[str, Any]


class AdaptBody(BaseModel):
    # 雛形のままの定義, 雛形の名前, 引数の名前ごとの値
    spec: dict[str, Any]
    name: str
    values: dict[str, Any]


def loose_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    定義から照らさない項目を除いた写しを返す関数

    Parameters
    ----------
    spec: dict[str, Any]
        定義

    Returns
    ----------
    loose: dict[str, Any]
        位置, 名前と説明, 入力の形と型と, 空の項目を除いた写し
    """
    # 上の段の位置, 名前, 説明と空の項目
    loose = {
        key: copy.deepcopy(value) for key, value in spec.items()
        if key not in LOOSE_KEYS and value not in (None, {}, [])
    }

    # 入力の位置, 形, 型と, ノードの位置と空の項目
    if isinstance(loose.get("inputs"), dict):
        loose["inputs"] = {
            name: drop_keys(value, LOOSE_INPUT_KEYS)
            for name, value in loose["inputs"].items()
        }
    if isinstance(loose.get("nodes"), dict):
        loose["nodes"] = {
            name: drop_keys(value, ("ui",))
            for name, value in loose["nodes"].items()
        }

    return loose


def drop_keys(value: Any, keys: tuple[str, ...]) -> Any:
    """
    辞書から鍵と空の項目を除いた写しを返す関数

    Parameters
    ----------
    value: Any
        入力かノードの項目 (辞書でなければそのまま)
    keys: tuple[str, ...]
        除く鍵

    Returns
    ----------
    dropped: Any
        鍵と, 値が None か空の辞書の項目を除いた写し
    """
    # 辞書でない値はそのまま
    if not isinstance(value, dict):
        return value
    dropped = {
        key: item for key, item in value.items()
        if key not in keys and item not in (None, {})
    }

    return dropped


def place_model(place: tuple[str, str, Any, int]) -> TemplatePlace:
    """
    書き換える場所を API の形にする関数

    Parameters
    ----------
    place: tuple[str, str, Any, int]
        (種類, 名前, 鍵, 倍率)

    Returns
    ----------
    model: TemplatePlace
        API の形の場所
    """
    kind, owner, key, scale = place
    model = TemplatePlace(kind=kind, owner=owner, key=key, scale=scale)

    return model


def current_args(spec: dict[str, Any], name: str) -> Optional[dict[str, int]]:
    """
    定義が雛形のままなら, 書き換える引数の今の値を返す関数

    引数の最初の場所の値を倍率で割って今の値とし, その値で adapt_example
    した定義と, 照らさない項目を除いて同じなら雛形のままとみなす

    Parameters
    ----------
    spec: dict[str, Any]
        定義
    name: str
        雛形の名前 (TEMPLATE_ARGS にある同梱の例)

    Returns
    ----------
    values: Optional[dict[str, int]]
        引数の名前ごとの今の値 (雛形のままでなければ None)
    """
    # 最初の場所の今の値 (読めないか, 倍率で割り切れない 1 未満は雛形でない)
    places = template_places(name)
    values: dict[str, int] = {}
    for arg, found in places.items():
        try:
            raw = read_place(spec, found[0])
        except READ_ERRORS:
            return None
        scale = found[0][3]
        if raw < 1 or raw % scale != 0:
            return None
        values[arg] = raw // scale

    # 今の値で書き換えた雛形との照らし合わせ
    expected = adapt_example(name, values)
    if loose_spec(spec) != loose_spec(expected):
        return None

    return values


def find_template(spec: dict[str, Any]) -> Optional[TemplateMatch]:
    """
    定義が書き換えられる雛形のままかを照らし, 雛形と引数を返す関数

    Parameters
    ----------
    spec: dict[str, Any]
        定義

    Returns
    ----------
    match: Optional[TemplateMatch]
        雛形の名前と, 引数ごとの今の値と場所 (雛形でなければ None)
    """
    # 雛形の順に照らし, 最初に合ったもの
    for name, places in TEMPLATE_ARGS.items():
        values = current_args(spec, name)
        if values is None:
            continue
        args = [
            TemplateArg(
                name=arg, value=values[arg],
                places=[place_model(place) for place in found]
            )
            for arg, found in places.items()
        ]
        match = TemplateMatch(name=name, args=args)
        return match

    return None


def kept_spec(
    adapted: dict[str, Any], spec: dict[str, Any], name: str
) -> dict[str, Any]:
    """
    書き換えた雛形に, 今の定義の位置, 名前と説明, 入力の形と型を残す関数

    Parameters
    ----------
    adapted: dict[str, Any]
        adapt_example が返した定義 (書き換える)
    spec: dict[str, Any]
        今の定義
    name: str
        雛形の名前

    Returns
    ----------
    kept: dict[str, Any]
        書き換える軸以外の入力の形と型, 位置, 名前と説明を今の定義にした
        定義
    """
    # 上の段の位置, 名前, 説明 (今の定義に無ければ消す)
    kept = adapted
    for key in LOOSE_KEYS:
        if key in spec:
            kept[key] = copy.deepcopy(spec[key])
        else:
            kept.pop(key, None)

    # ノードの位置
    for node, value in kept["nodes"].items():
        before = spec["nodes"][node]
        if isinstance(before, dict) and "ui" in before:
            value["ui"] = copy.deepcopy(before["ui"])
        else:
            value.pop("ui", None)

    # 入力の項目は今の定義のまま, 形は書き換える軸だけ雛形の値
    axes = input_axes(name)
    for item, value in kept["inputs"].items():
        before = copy.deepcopy(spec["inputs"][item])
        shape = before.get("shape")
        written = value.get("shape")
        if isinstance(shape, list) and isinstance(written, list) and (
            len(shape) == len(written)
        ):
            for axis in axes.get(item, []):
                shape[axis] = written[axis]
        else:
            before["shape"] = written
        kept["inputs"][item] = before

    return kept


def input_axes(name: str) -> dict[str, list[int]]:
    """
    雛形の, 入力ごとの書き換える軸を返す関数

    Parameters
    ----------
    name: str
        雛形の名前

    Returns
    ----------
    axes: dict[str, list[int]]
        入力の名前ごとの軸
    """
    # 種類 input の場所
    axes: dict[str, list[int]] = {}
    for found in template_places(name).values():
        for kind, owner, key, _ in found:
            if kind == "input":
                axes.setdefault(owner, []).append(key)

    return axes


@router.post("/templates/match")
def match_template(body: MatchBody) -> Optional[TemplateMatch]:
    """
    定義が書き換えられる雛形のままかを照らす関数

    Parameters
    ----------
    body: MatchBody
        照らす定義

    Returns
    ----------
    match: Optional[TemplateMatch]
        雛形の名前と, 引数ごとの今の値と場所 (雛形でなければ None)
    """
    match = find_template(body.spec)

    return match


@router.post("/templates/adapt")
def adapt_template(body: AdaptBody) -> TemplateAdapted:
    """
    雛形のままの定義を, 渡した引数の値に書き換える関数

    Parameters
    ----------
    body: AdaptBody
        雛形のままの定義, 雛形の名前, 引数の名前ごとの値 (書かない引数は
        今の定義の値のまま)

    Returns
    ----------
    adapted: TemplateAdapted
        書き換えた定義と, 渡した引数ごとの前と後の値と場所
    """
    # 書き換えられる雛形 (無ければ 404) と, 雛形のままか (でなければ 409)
    try:
        places = template_places(body.name)
    except KeyError as error:
        raise HTTPException(
            404, f"{body.name} は書き換えられる雛形ではありません"
        ) from error
    current = current_args(body.spec, body.name)
    if current is None:
        raise HTTPException(
            409, f"定義が雛形 {body.name} のままではないので書き換えません"
        )

    # 今の値に渡した値を重ねた書き換え (値の誤りは 422)
    try:
        spec = adapt_example(body.name, {**current, **body.values})
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    spec = kept_spec(spec, body.spec, body.name)

    # 渡した引数ごとの前と後の値
    changes = [
        TemplateChange(
            name=arg, before=current[arg],
            after=read_place(spec, places[arg][0]) // places[arg][0][3],
            places=[place_model(place) for place in places[arg]]
        )
        for arg in body.values
    ]
    adapted = TemplateAdapted(name=body.name, spec=spec, args=changes)

    return adapted
