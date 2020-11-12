"""デフォルトボックスを作成する
"""
import itertools
import math

import torch


# 4種類： 小さい正方形, 大きい正方形、 1:2の長方形, 2:1の長方形
# 6種類： 小さい正方形, 大きい正方形、 1:2の長方形, 2:1の長方形, 1:3の長方形, 3:1の長方形
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 初期設定
        self.image_size = cfg['input_size']  # 画像サイズ 300 x 300の300
        self.feature_maps = cfg['feature_maps']  # [38, 19, 10, 5, 3, 1] 各sourceの特徴量マップのサイズ
        self.num_priors = len(cfg['feature_maps'])  # sourceの個数=6
        self.steps = cfg['steps']  # [8, 16, 32, 64, 100, 300]    DBoxのピクセルサイズ
        self.min_sizes = cfg['min_sizes']  # [30, 60, 111, 162, 213, 264] 小さい正方形のDBoxのピクセルサイズ
        self.max_sizes = cfg['max_sizes']  # [60, 111, 162, 213, 264, 315] 大きい正方形のDBoxのピクセルサイズ
        self.aspect_ratios = cfg['aspect_ratios']  # 長方形のDBoxのアスペクト比


    def make_dbox_list(self):
        '''DBoxを作成'''
        mean = []

        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):  # 重複ありのfまでの数で2ペアの組み合わせ
                # product(range(3), repeat=2)
                # = (0, 0), (0, 1), (0, 2)
                #   (1, 0), (1, 1), (1, 2)
                #   (2, 0), (2, 1), (2, 2)

                # 特徴マップの1要素が1つのDBoxに対応することが前提
                # 'steps': [8, 16, 32, 64, 100, 300] # DBoxのサイズ
                # 300 / steps = [37.5, 18.75, 9.375, 4.6875, 3, 1]
                f_k = self.image_size / self.steps[k]

                # DBoxの中心座標 (x, y) ただし、0~1で規格化
                # e.g (37.5, 18.5) / 37.5 = (1.0, 0.4933..)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # アスペクト比1の小さいDBox[cx, cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # アスペクト比1の大きいDBox[cx, cy, width, height]
                # 'max_sizes': [60, 111, 162, 213, 264, 315]
                s_k_prime = math.sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # その他のアスペクト比のdefBox [cx, cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]  # 横長長方形
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]  # 縦長長方形

                # DBoxをテンソルに変換 torch.Size([8732, 4])
                output = torch.Tensor(mean).view(-1, 4)

                # DBoxが画像の外にはみ出るのを防ぐために、大きさを最小0, 最大1にする
                output.clamp_(max=1, min=0)

        return output


if __name__ == "__main__":

    # SSD300の設定
    SSD300_cfg = {
        'num_classes': 21,  # 背景クラスを含めた合計クラス数
        'input_size' : 300, # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4,],    # 出力するDBoxのアスペクト比の種類
        'feature_maps': [38, 19, 10, 5, 3, 1],     # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300] ,       # DBoxのピクセルサイズ
        'min_sizes': [30, 60, 111, 162, 213, 264], # 小さい正方形のDBoxのピクセルサイズ
        'max_sizes': [45, 99, 153, 207, 261, 315], # 大きい正方形のDBoxのピクセルサイズ
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]] # アスペクト比の構成?
    }

    # DBoxの作成
    dbox = DBox(SSD300_cfg)
    dbox_list = dbox.make_dbox_list()

    # DBoxの出力を確認
    import pandas as pd
    dbox_df = pd.DataFrame(dbox_list)
    print(dbox_df)