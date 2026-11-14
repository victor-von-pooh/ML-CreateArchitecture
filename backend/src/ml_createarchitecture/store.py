import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ml_modelcontrol import dump_spec, load_spec

# モデルの定義のファイルの名前 (拡張子を除く) に使える文字
NAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}")


class ModelError(ValueError):
    pass


# 定義が無い場合の誤り (消す場合に, 消せない誤りと分けるため)
class ModelNotFoundError(ModelError):
    pass


class ModelStore:
    def __init__(self, root: Path):
        # 定義のファイルを置くディレクトリを保存 (書き込むときに作成)
        self.root = root

    def path_for(self, name: str) -> Path:
        """
        名前から定義のファイルのパスを求める関数

        Parameters
        ----------
        name: str
            定義の名前 (英数字, _, - で 64 文字まで)

        Returns
        ----------
        path: Path
            置き場の中の YAML のファイルのパス

        Raises
        ----------
        ModelError
            名前の規則に合わない場合
        """
        # 置き場の外を指せないように名前の文字を限定
        if not NAME_PATTERN.fullmatch(name):
            raise ModelError(
                f"名前 {name!r} は英数字, _, - (64 文字まで) にしてください"
            )
        path = self.root / f"{name}.yaml"

        return path

    def list_models(self) -> list[dict[str, Any]]:
        """
        置き場の定義のファイルの一覧を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        models: list[dict[str, Any]]
            名前と更新時刻 (新しい順)
        """
        # 置き場が無ければ空, ある場合は規則に合う名前のファイルだけを一覧に
        models: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return models
        for path in self.root.glob("*.yaml"):
            if not NAME_PATTERN.fullmatch(path.stem) or not path.is_file():
                continue
            updated = datetime.fromtimestamp(path.stat().st_mtime)
            models.append({
                "name": path.stem,
                "updated_at": updated.isoformat(timespec="seconds")
            })
        models.sort(key=lambda item: item["updated_at"], reverse=True)

        return models

    def load(self, name: str) -> dict[str, Any]:
        """
        定義を読み込む関数

        Parameters
        ----------
        name: str
            定義の名前

        Returns
        ----------
        spec: dict[str, Any]
            モデルの定義

        Raises
        ----------
        ModelError
            ファイルが無いか, 読めない場合
        """
        # ファイルの存在と, 定義として読めることの確認
        path = self.path_for(name)
        if not path.is_file():
            raise ModelError(f"定義 {name} はありません")
        try:
            spec: dict[str, Any] = load_spec(path)
        except (ValueError, OSError, yaml.YAMLError) as error:
            raise ModelError(f"定義 {name} を読めません: {error}") from error

        return spec

    def save(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """
        定義を保存する関数

        途中で止まっても壊れたファイルが残らないように, 一時ファイルに書いて
        から置き換える

        Parameters
        ----------
        name: str
            定義の名前
        spec: dict[str, Any]
            モデルの定義

        Returns
        ----------
        summary: dict[str, Any]
            保存した定義の名前と更新時刻

        Raises
        ----------
        ModelError
            名前の規則に合わない場合, 版を読めない場合, 書き込めない場合
        """
        # 版を確かめてから YAML に変換
        path = self.path_for(name)
        try:
            text: str = dump_spec(load_spec(spec))
        except ValueError as error:
            raise ModelError(str(error)) from error

        # 書き手ごとに別の一時ファイルに書いてから置き換え
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(
                dir=self.root, prefix=f".{name}.", suffix=".tmp"
            )
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(text)
            Path(temp).replace(path)
            updated = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError as error:
            raise ModelError(f"定義 {name} を保存できません: {error}") from error
        summary = {
            "name": name, "updated_at": updated.isoformat(timespec="seconds")
        }

        return summary

    def delete(self, name: str) -> None:
        """
        保存した定義を消す関数

        Parameters
        ----------
        name: str
            定義の名前

        Returns
        ----------
        None

        Raises
        ----------
        ModelError
            名前の規則に合わない場合, ファイルが無い場合, 消せない場合
        """
        # 名前の規則で置き場の中に限定し, 無い場合は誤り
        path = self.path_for(name)
        try:
            path.unlink()
        except FileNotFoundError as error:
            raise ModelNotFoundError(f"定義 {name} はありません") from error
        except OSError as error:
            raise ModelError(f"定義 {name} を消せません: {error}") from error
