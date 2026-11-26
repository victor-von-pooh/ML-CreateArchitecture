"""
画面の通しのテスト用に, 書き出したデータセット, 学習のプロジェクトと,
ML-Pathfinder の置き場の問題の設定を作るスクリプト (ML-EDA と ML-Quest で
作る. 追加の依存 project が要る)
"""

import hashlib
import sys
from pathlib import Path

import ml_eda
import ml_quest
import numpy as np
import pandas as pd
import yaml

# 異常の検知の表の列 (行の id, 度合いに使う 3 列, 異常の正解)
READINGS = ["reading_id", "temp", "vibration", "current", "is_anomaly"]

# 介入の記録の表の列 (行の id, 効果の見積もりに使う 2 列, 処置, 結果)
COUPONS = ["customer_id", "age", "visits", "coupon", "purchased"]

# 値段を変えた記録の表の列 (行の id, 商品の id, 売れ行きの予測に使う 2 列,
# 値段, 売れた数, 原価)
SALES = ["record_id", "product_id", "size", "weekday", "price", "units",
         "unit_cost"]

# 需要の記録の表の列 (行の id, 商品の id, 日付, 商品の特徴, 日ごとの売れた
# 数, 在庫の数)
DEMANDS = ["record_id", "product_id", "date", "size", "units", "stock"]


def make_dataset(path: Path) -> None:
    """
    3 列の 3 クラスの分類の表を, データセットとして書き出す関数

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 決まった乱数の表を CSV にして, ML-EDA で読んで書き出し
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "a": rng.normal(size=60), "b": rng.normal(size=60),
        "c": rng.normal(size=60),
        "label": np.array(["x", "y", "z"] * 20)
    })
    source = path.parent / "table.csv"
    frame.to_csv(source, index=False)
    recipe = {
        "target": "label", "task": "classification", "steps": [],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2}
    }
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, path)


def make_masks(path: Path) -> None:
    """
    灰色の画像と 3 つの値の領域の, ファイルの組のデータセット (版 3) の
    説明を書く関数 (model_inputs は説明だけを読むので分割のファイルは
    書かない)

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 大きさの混ざる画像 (高さと幅は決まらない) と, 背景 0 の名前の無い領域
    path.mkdir(parents=True)
    info = {
        "format_version": 3, "role": "predict",
        "target": {"kind": "mask", "entries": ["mask"]},
        "entries": {
            "image": {"kind": "image", "role": "input",
                      "shape": [None, None, 1], "dtype": "uint8",
                      "steps": ["grayscale"]},
            "mask": {"kind": "mask", "role": "target",
                     "shape": [None, None], "dtype": "uint8",
                     "values": [0, 1, 2],
                     "classes": [{"value": 1, "name": "circle"},
                                 {"value": 2, "name": "square"}]},
        },
    }
    (path / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True), encoding="utf-8"
    )


def make_clouds(path: Path) -> None:
    """
    128 点の観測の点群から 64 点の点群を当てる, ファイルの組のデータセット
    (版 2) の説明を書く関数 (model_inputs は説明だけを読むので分割の
    ファイルは書かない)

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 点の数のそろった観測の点群と正解の点群
    path.mkdir(parents=True)
    info = {
        "format_version": 2, "role": "predict",
        "target": {"kind": "point_cloud", "entries": ["cloud"]},
        "entries": {
            "observed": {"kind": "point_cloud", "role": "input",
                         "shape": [128, 3], "dtype": "float32"},
            "cloud": {"kind": "point_cloud", "role": "target",
                      "shape": [64, 3], "dtype": "float32"},
        },
    }
    (path / "dataset.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True), encoding="utf-8"
    )


def make_project(path: Path) -> None:
    """
    ML-Quest の例の問題の設定から, 学習のプロジェクトの骨組みを書き出す
    関数

    Parameters
    ----------
    path: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 最初の例 (列の予測) の骨組み
    problem = ml_quest.example_problem(next(iter(ml_quest.EXAMPLES)))
    ml_quest.make_project(problem, path)


def rank_problem() -> dict:
    """
    見本の課題から, 並べる受け持ちを一から学習する問題の設定 (受け持ちに
    合う雛形は two_tower_retrieval) を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        問題の設定 (反応の表の列は member, item_id, ts)
    """
    # 見本の受け持ちを作り直す
    problem = ml_quest.example_problem(next(iter(ml_quest.EXAMPLES)))
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = ml_quest.answer(
        problem, "information.columns", ["member", "item_id", "ts"]
    )
    answers = [
        ("role", "rank"), ("what", "試しの出力"), ("item", "1 件"),
        ("baseline", "今のやり方"), ("evaluation.link", "良くなると損が減る"),
        ("unit", "query"), ("output", "ranking"),
        ("target.source", "feedback"), ("method", "scratch"),
        ("evaluation.metric", "ndcg"), ("rank.user", "member"),
        ("rank.item", "item_id"), ("rank.time", "ts"), ("rank.k", 5)
    ]
    for key, value in answers:
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)
    problem["name"] = "e2e の推薦"

    return problem


def anomaly_problem(name: str, score: str) -> dict:
    """
    見本の課題から, 異常の検知の受け持ちを一から学ぶ問題の設定を作る関数

    Parameters
    ----------
    name: str
        問題の設定の名前
    score: str
        度合いの決め方 (autoencoder なら雛形は tabular_autoencoder,
        distance なら雛形は無し)

    Returns
    ----------
    problem: dict
        問題の設定 (度合いに使う列は temp, vibration, current. 異常の正解の
        列は is_anomaly, 行の id の列は reading_id)
    """
    # 見本の受け持ちの作り直し (ML-Quest のテストの受け持ちの入れ替えと同じ)
    problem = ml_quest.example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = ml_quest.answer(problem, "information.columns", READINGS)
    answers = [
        ("role", "predict"), ("unit", "row"), ("output", "value"),
        ("target.source", "few"), ("method", "scratch"),
        ("target.kind", "anomaly"), ("what", "機械の異常の度合い"),
        ("item", "1 回の計測"), ("baseline", "決まった値を超えたら"),
        ("anomaly.label", "is_anomaly"), ("anomaly.id", "reading_id"),
        ("anomaly.columns", ["temp", "vibration", "current"]),
        ("anomaly.score", score), ("anomaly.capacity", 5)
    ]
    if score == "autoencoder":
        answers += [("evaluation.metric", "average_precision"),
                    ("evaluation.link", "上位に異常が多いほど見逃しが減る")]
    for key, value in answers:
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)
    problem["name"] = name

    return problem


def make_readings(path: Path) -> None:
    """
    センサの 3 列と, 異常の行にだけ印のある正解の列の表を, 異常の検知の表
    (task: anomaly) として書き出す関数

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 学習に使う列から外す正解の列と行の id の列 (学習に使う列は 3)
    rng = np.random.default_rng(0)
    rows = 40
    frame = pd.DataFrame({
        "reading_id": [f"r{number:02d}" for number in range(rows)],
        "temp": rng.normal(size=rows), "vibration": rng.normal(size=rows),
        "current": rng.normal(size=rows),
        "is_anomaly": [1 if number >= rows - 4 else None
                       for number in range(rows)]
    })
    source = path.parent / "readings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source, index=False)
    recipe = {
        "task": "anomaly", "target": "is_anomaly", "id": "reading_id",
        "steps": [],
        "split": {"method": "random", "valid": 0.2, "test": 0.2,
                  "unlabeled": "normal"}
    }
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, path)


def uplift_problem() -> dict:
    """
    見本の課題から, 今ある記録から効果を見積もる問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        効果を見積もるで方策でなく, 今ある記録から集める問題の設定 (処置の
        列は coupon, 結果の列は purchased, 行の id の列は customer_id,
        効果の見積もりに使う列は age と visits)
    """
    # 見本の受け持ちの作り直し (ML-Quest のテストの受け持ちの入れ替えと同じ)
    problem = ml_quest.example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = ml_quest.answer(problem, "information.columns", COUPONS)
    answers = [
        ("role", "effect"), ("unit", "row"), ("output", "effect"),
        ("target.source", "intervention"), ("method", "scratch"),
        ("autonomous", False), ("collect", "log"),
        ("what", "クーポンで買う確率がどれだけ上がるか"),
        ("item", "顧客ごとの 1 行"), ("baseline", "全員に配る"),
        ("uplift.treatment", "coupon"), ("uplift.outcome", "purchased"),
        ("uplift.outcome_kind", "binary"), ("uplift.id", "customer_id"),
        ("uplift.columns", ["age", "visits"]), ("uplift.k", 0.3),
        ("evaluation.metric", "qini"),
        ("evaluation.link", "効果の順が当たるほど配る費用が実る")
    ]
    for key, value in answers:
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)
    problem = ml_quest.answer(
        problem, "action.decision", {"type": "treat", "budget": 10}
    )
    problem["name"] = "e2e のクーポンの効果"

    return problem


def make_coupons(path: Path) -> None:
    """
    無作為にクーポンを配った記録を, 介入の記録の表 (task: uplift) として
    書き出す関数

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 学習に使う列から外す列 (処置の列, 結果の列, 行の id の列. 学習に使う
    # 列は 2)
    rng = np.random.default_rng(0)
    rows = 40
    frame = pd.DataFrame({
        "customer_id": [f"c{number:02d}" for number in range(rows)],
        "age": rng.uniform(20, 60, size=rows),
        "visits": rng.uniform(0, 10, size=rows),
        "coupon": [number % 2 for number in range(rows)],
        "purchased": rng.integers(0, 2, size=rows)
    })
    source = path.parent / "coupons.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source, index=False)
    recipe = {
        "task": "uplift", "target": "purchased", "treatment": "coupon",
        "id": "customer_id", "steps": [],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2}
    }
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, path)


def price_problem() -> dict:
    """
    見本の課題から, 試しに値段を変えた記録から値段を選ぶ問題の設定を作る
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        効果を見積もるで方策でなく, 今ある記録で値段を変えた問題の設定 (値段
        の列は price, 売れた数の列は units, 原価の列は unit_cost, 商品の id
        の列は product_id, 行の id の列は record_id, 売れ行きの予測に使う
        列は size と weekday)
    """
    # 見本の受け持ちの作り直し (ML-Quest のテストの受け持ちの入れ替えと同じ)
    problem = ml_quest.example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = ml_quest.answer(problem, "information.columns", SALES)
    answers = [
        ("role", "effect"), ("unit", "row"), ("output", "effect"),
        ("target.source", "intervention"), ("method", "scratch"),
        ("autonomous", False), ("collect", "log"), ("varied", "price"),
        ("what", "値段を変えたときの売れた数"),
        ("item", "商品と日ごとの 1 行"), ("baseline", "今の値段のまま"),
        ("price.price", "price"), ("price.units", "units"),
        ("price.cost", "unit_cost"), ("price.product", "product_id"),
        ("price.id", "record_id"), ("price.columns", ["size", "weekday"]),
        ("evaluation.metric", "mae"),
        ("evaluation.link", "売れた数の外れが小さいほど利益の見込みが当たる")
    ]
    for key, value in answers:
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)
    problem = ml_quest.answer(problem, "action.decision", {
        "type": "price", "candidates": [-0.1, 0, 0.1], "max_change": 0.1
    })
    problem["name"] = "e2e の値段の見直し"

    return problem


def make_sales(path: Path) -> None:
    """
    試しに値段を変えた売れ行きの記録を, 値段を変えた記録の表 (task: price)
    として書き出す関数

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 学習に使う列から外す列 (売れた数, 原価, 商品の id, 行の id の列. 学習
    # に使う列は特徴の 2 列と最後の値段の 1 列)
    rows = 40
    frame = pd.DataFrame({
        "record_id": [f"r{number:03d}" for number in range(rows)],
        "product_id": [f"p{number % 4}" for number in range(rows)],
        "size": [float(number % 4) for number in range(rows)],
        "weekday": [number % 7 for number in range(rows)],
        "price": [
            100 * (1 + 0.1 * (number % 5 - 2)) for number in range(rows)
        ],
        "units": [20 - number % 5 for number in range(rows)],
        "unit_cost": 60.0
    })
    source = path.parent / "sales.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source, index=False)
    recipe = {
        "task": "price", "target": "units", "price": "price",
        "cost": "unit_cost", "product": "product_id", "id": "record_id",
        "steps": [{"op": "standardize", "columns": "auto"}],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2}
    }
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, path)


def demand_problem() -> dict:
    """
    見本の課題から, 商品ごとの日ごとの売れた数の記録から発注の数を決める
    問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        予測するで正解の種類が先の期間の需要の問題の設定 (商品の id の列は
        product_id, 日付の列は date, 売れた数の列は units, 在庫の数の列は
        stock, 行の id の列は record_id, 商品の特徴の列は size. 学ぶ水準の
        答えは 0.1 と 0.9 で, 品切れの損 3 と売れ残りの損 1 の水準 τ 0.75
        と 0.5 を足した 4 つを学ぶ)
    """
    # 見本の受け持ちの作り直し (ML-Quest のテストの受け持ちの入れ替えと同じ)
    problem = ml_quest.example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = ml_quest.answer(problem, "information.columns", DEMANDS)
    answers = [
        ("role", "predict"), ("unit", "row"), ("output", "distribution"),
        ("target.kind", "demand"), ("target.source", "column"),
        ("method", "scratch"), ("what", "先の 7 日の売れる数"),
        ("item", "商品と発注を決める日ごとの 1 行"),
        ("baseline", "前の 7 日と同じ数を発注する"),
        ("demand.product", "product_id"), ("demand.date", "date"),
        ("demand.units", "units"), ("demand.stock", "stock"),
        ("demand.id", "record_id"), ("demand.columns", ["size"]),
        ("demand.levels", [0.1, 0.9]),
        ("evaluation.metric", "loss"),
        ("evaluation.link", "分位点の損失が小さいほど発注の数の損の見込みが"
         "当たる")
    ]
    for key, value in answers:
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)
    problem = ml_quest.answer(problem, "action.decision", {
        "type": "order", "shortage_cost": 3, "holding_cost": 1,
        "max_order": 60
    })
    problem["name"] = "e2e の発注の見直し"

    return problem


def make_demands(path: Path) -> None:
    """
    商品ごとの日ごとの売れた数の記録を, 需要の記録の表 (task: demand) と
    して書き出す関数

    Parameters
    ----------
    path: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 学習に使う列は, 商品の特徴の 1 列, 過去の売れた数の 1 日前と 7 日前と
    # 7 日の移動平均, 曜日の 5 列 (売れた数, 商品の id, 日付, 在庫の数, 行の
    # id の列は数えない)
    days, products = 84, 3
    rows = days * products
    frame = pd.DataFrame({
        "record_id": [f"r{number:04d}" for number in range(rows)],
        "product_id": [f"p{number // days}" for number in range(rows)],
        "date": list(pd.date_range("2024-01-01", periods=days, freq="D"))
        * products,
        "size": [float(number // days) for number in range(rows)],
        "units": [5 + number % 7 for number in range(rows)],
        "stock": 3.0
    })
    source = path.parent / "demands.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source, index=False)
    recipe = {
        "task": "demand", "target": "units", "product": "product_id",
        "date": "date", "stock": "stock", "horizon": 7, "id": "record_id",
        "steps": [
            {"op": "lag", "columns": ["units"], "lags": [1, 7],
             "windows": [7], "by": ["product_id"], "time": "date",
             "freq": "1D"},
            {"op": "datetime_parts", "columns": ["date"],
             "parts": ["weekday"]},
            {"op": "standardize", "columns": "auto"}
        ],
        "split": {"method": "time", "valid": 0.2, "test": 0.2}
    }
    ml_eda.export_dataset(ml_eda.load_data(source), recipe, path)


def make_store(
    path: Path, project: Path, dataset: Path, rank_project: Path,
    anomaly_project: Path, distance_project: Path, uplift_project: Path,
    price_project: Path, demand_project: Path
) -> None:
    """
    ML-Pathfinder の置き場に, プロジェクトを書き出した問題の設定と, まだ
    書き出していない問題の設定を保存する関数

    Parameters
    ----------
    path: Path
        ML-Pathfinder の置き場
    project: Path
        書き出したプロジェクトのディレクトリ
    dataset: Path
        記録に書くデータセットのディレクトリ
    rank_project: Path
        雛形のある問題の設定のプロジェクトのディレクトリ (並べるの骨組みを
        書き出し, 反応の表を置く)
    anomaly_project: Path
        異常の検知の問題の設定のプロジェクトのディレクトリ (自己符号化器の
        骨組みを書き出し, 記録の場所に異常の検知の表を書き出す)
    distance_project: Path
        異常の検知の距離に基づく度合いの問題の設定のプロジェクトの
        ディレクトリ
    uplift_project: Path
        効果の見積もりの問題の設定のプロジェクトのディレクトリ (効果の
        見積もりの骨組みを書き出し, 記録の場所に介入の記録の表を書き出す)
    price_project: Path
        値段を変える判断の問題の設定のプロジェクトのディレクトリ (値段を
        変える判断の骨組みを書き出し, 記録の場所に値段を変えた記録の表を
        書き出す)
    demand_project: Path
        時間をまたぐ量の判断の問題の設定のプロジェクトのディレクトリ (時間を
        またぐ量の判断の骨組みを書き出し, 記録の場所に需要の記録の表を
        書き出す)

    Returns
    ----------
    None
    """
    # 書き出した方は, 記録のデータセットを作ったデータセットにする (入力と
    # 出力を作れるように)
    problem = ml_quest.example_problem(next(iter(ml_quest.EXAMPLES)))
    recorded = ml_quest.record_project(
        {**problem, "name": "e2e の売上"}, project
    )
    recorded["project"]["dataset"] = str(dataset.resolve())
    ml_quest.save_problem(recorded, path / "problems" / "recorded")
    draft = {**problem, "name": "e2e の下書き"}
    ml_quest.save_problem(draft, path / "problems" / "draft")

    # 書き出してあり, モデルの定義の置き場の記録だけが無いもの
    no_spec = ml_quest.record_project(
        {**problem, "name": "e2e の定義の記録なし"}, project
    )
    no_spec["project"]["model_spec"] = None
    ml_quest.save_problem(no_spec, path / "problems" / "no_spec")

    # 受け持ちに合う雛形があり, 保存先にまだ定義が無いもの (別のプロジェクト.
    # 他のテストが書く定義と重ならないように). 反応の表は利用者 6 人と品物
    # 4 つの 40 行で, 新しい 4 行だけに出る利用者 m9 と品物 i9 は, 骨組みの
    # 割合 (検証用とテスト用が 1 割ずつ) では学習用にいない
    ml_quest.make_project(rank_problem(), rank_project)
    ranked = ml_quest.record_project(rank_problem(), rank_project)
    rows = [
        {"member": f"m{number % 6}", "item_id": f"i{number % 4}",
         "ts": number}
        for number in range(36)
    ] + [{"member": "m9", "item_id": "i9", "ts": 36 + number}
         for number in range(4)]
    reactions = Path(ranked["project"]["dataset"])
    reactions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(reactions, index=False)
    ml_quest.save_problem(ranked, path / "problems" / "rank")

    # 異常の検知の問題の設定 (自己符号化器と距離に基づく度合い. 他のテストが
    # 書く定義と重ならないように別のプロジェクト)
    for name, score, place in (
        ("e2e の異常の検知", "autoencoder", anomaly_project),
        ("e2e の距離の度合い", "distance", distance_project),
    ):
        anomaly = anomaly_problem(name, score)
        ml_quest.make_project(anomaly, place)
        recorded = ml_quest.record_project(anomaly, place)
        if score == "autoencoder":
            make_readings(Path(recorded["project"]["dataset"]))
        ml_quest.save_problem(recorded, path / "problems" / score)

    # 効果の見積もりの問題の設定 (他のテストが書く定義と重ならないように
    # 別のプロジェクト)
    ml_quest.make_project(uplift_problem(), uplift_project)
    recorded = ml_quest.record_project(uplift_problem(), uplift_project)
    make_coupons(Path(recorded["project"]["dataset"]))
    ml_quest.save_problem(recorded, path / "problems" / "uplift")

    # 値段を変える判断の問題の設定 (他のテストが書く定義と重ならないように
    # 別のプロジェクト)
    ml_quest.make_project(price_problem(), price_project)
    recorded = ml_quest.record_project(price_problem(), price_project)
    make_sales(Path(recorded["project"]["dataset"]))
    ml_quest.save_problem(recorded, path / "problems" / "price")

    # 時間をまたぐ量の判断の問題の設定 (他のテストが書く定義と重ならない
    # ように別のプロジェクト)
    ml_quest.make_project(demand_problem(), demand_project)
    recorded = ml_quest.record_project(demand_problem(), demand_project)
    make_demands(Path(recorded["project"]["dataset"]))
    ml_quest.save_problem(recorded, path / "problems" / "demand")


def make_experiment(project: Path) -> None:
    """
    プロジェクトの実験の置き場に, 定義の写しを持つ ML-Found の形の実験を
    作る関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 実験の置き場は config の experiment.output_dir (無ければ outputs)
    config = yaml.safe_load(
        (project / "configs" / "default" / "train.yaml").read_text()
    )
    root = (config.get("experiment") or {}).get("output_dir") or "outputs"
    run_dir = project / root / "e2e_run" / "2026-11-20" / "10-00-00"
    run_dir.mkdir(parents=True)
    spec = {
        "format_version": 1,
        "name": "e2e_run",
        "inputs": {"x": {"shape": [4, 3]}},
        "nodes": {
            "e2e_fc": {
                "unit": "nn.Linear",
                "params": {"in_features": 3, "out_features": 2},
                "inputs": "x",
            }
        },
        "outputs": {"y": "e2e_fc"},
    }
    data = yaml.safe_dump(spec, sort_keys=False).encode()
    (run_dir / "model_spec.yaml").write_bytes(data)
    (run_dir / "config.yaml").write_text("model:\n  spec: models/x.yaml\n")
    record = {
        "path": "models/e2e_run.yaml", "status": "copied",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (run_dir / "source.yaml").write_text(
        yaml.safe_dump({"format_version": 1, "model_spec": record})
    )
    (run_dir / "run.json").write_text('{"status": "completed"}')


if __name__ == "__main__":
    work = Path(sys.argv[1])
    make_dataset(work / "dataset")
    make_masks(work / "masks")
    make_clouds(work / "clouds")
    make_project(work / "project")
    make_experiment(work / "project")
    make_store(
        work / "pathfinder", work / "project", work / "dataset",
        work / "rank_project", work / "anomaly_project",
        work / "distance_project", work / "uplift_project",
        work / "price_project", work / "demand_project"
    )
