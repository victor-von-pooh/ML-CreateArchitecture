import os
from pathlib import Path

# アプリのデータの置き場を変える環境変数
HOME_ENV = "ML_CREATEARCHITECTURE_HOME"

# 起動のときに読み込む自作の部品のモジュール (カンマ区切り) の環境変数
MODULES_ENV = "ML_CREATEARCHITECTURE_MODULES"

# ML-Pathfinder の置き場を変える環境変数 (ML-Pathfinder と同じ名前)
PATHFINDER_HOME_ENV = "ML_PATHFINDER_HOME"


def get_data_dir() -> Path:
    """
    アプリ自身のデータを置くディレクトリを取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    data_dir: Path
        モデルの定義のファイルを置く場所, 書き込むときに作成
    """
    # 環境変数があればそれを使い, 無ければホームディレクトリの下に設定
    value = os.environ.get(HOME_ENV)
    if value:
        data_dir = Path(value).expanduser().resolve()
    else:
        data_dir = Path.home() / ".ml-createarchitecture"

    return data_dir


def get_custom_modules() -> list[str]:
    """
    起動の設定に書いた自作の部品のモジュールの名前を取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    modules: list[str]
        モジュールの名前 (書いた順. 空白と空の要素と重なりは除く)
    """
    # カンマで区切り, 前後の空白を除いて並べる
    value = os.environ.get(MODULES_ENV, "")
    modules: list[str] = []
    for item in value.split(","):
        name = item.strip()
        if name and name not in modules:
            modules.append(name)

    return modules


def get_problems_dir() -> Path:
    """
    ML-Pathfinder が問題の設定を保存するディレクトリを取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    problems_dir: Path
        ML-Pathfinder の置き場の problems (無いこともある)
    """
    # ML-Pathfinder と同じ決め方 (環境変数か, ホームの下)
    value = os.environ.get(PATHFINDER_HOME_ENV)
    if value:
        home = Path(value).expanduser().resolve()
    else:
        home = Path.home() / ".ml-pathfinder"
    problems_dir = home / "problems"

    return problems_dir
