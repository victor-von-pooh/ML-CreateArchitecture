import torch
from torch import nn


class Shift(nn.Module):
    """入力に決めた値を足す部品"""

    def __init__(self, amount: float = 1.0):
        # 親クラスのコンストラクタを呼び出す
        super().__init__()

        # 足す値を保存
        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力に値を足す関数

        Parameters
        ----------
        x: torch.Tensor
            入力

        Returns
        ----------
        y: torch.Tensor
            足した値
        """
        y = x + self.amount

        return y


# 起動の設定から読み込む部品の宣言
UNITS = [{"cls": Shift, "name": "gui_lab.Shift"}]
