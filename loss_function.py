"""SSDの誤差関数の定義
"""

import torch
import torch.nn as nn
from torch.functional import F
# from make_match import match
from original_match import fnmatch

class MultiBoxLoss(nn.Module):
    """
    SSDの損失関数
    """
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super().__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = neg_pos
        self.device = device


    def forward(self, predictions, targets):
        """
        損失関数の計算
        Args:
            predictions: SSD netの訓練時の出力(tuple)
             loc=torch.Size([num_batch, 8732, 4]),
             conf=torch.Size([num_batch, 8732, 21]),
             dbox_list=torch.Size([8732, 4])

            targets: [num_batch, num_jobs, 5]
            5は正解アノテーション情報[xmin, ymin, xmax, ymax, label_index]を示す

        Returns:
            loss_l: locの損失値 SmoothL1Loss
            loss_c: confの損失値 CrossEntropyLoss
        """

        loc_data, conf_data, dbox_list = predictions
        # print("loc_data size: ", loc_data.size())
        num_batch = loc_data.size(0)  # ミニバッチ数(*)
        num_dbox = loc_data.size(1)  # DBox数(8732)
        num_classes = conf_data.size(2)  # クラス数(21)

        # 損失計算に使用する変数
        # conf_t_label: 各DBoxに、一番近い正解のBBoxのラベルを格納 8732
        # loc_t: 各DBoxに、一番近いBBoxのいち情報を格納 8732
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)  # torch.long
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)  # Tensorはtorch.float32
        # print("loc_t size: ", loc_t.size())
        # print("conf_t_label size: ", conf_t_label.size())

        # loc_tとconf_t_labelに, DBoxと正解アノテーションtargets(BBox)をmatchさせた結果を上書きする
        for idx in range(num_batch):
            truths_loc = targets[idx][:, :-1].to(self.device)  # BBox
            labels_conf = targets[idx][:, -1].to(self.device)  # Labels
            # print("truths_loc size: ", truths_loc.size())
            # print("labels_conf size: ", labels_conf)

            dbox = dbox_list.to(self.device)

            # 関数matchを実行し、loc_tとconf_t_labelの内容を更新する
            # (詳細)
            # loc_t: 各DBoxに、一番近い正解のBBoxの位置情報が上書きされる
            # conf_t_label: 各DBoxに、一番近い正解のBBoxのラベルが上書きされる
            # ただし、一番近いBBoxとのjaccard係数が0.5より小さい場合は、正解BBoxのconf_t_labelは背景クラス0とする
            variance = [0.1, 0.2]
            # loc_t[idx], conf_t_label[idx] = match(self.jaccard_thresh, truths_loc, dbox, variance, labels_conf)
            match(self.jaccard_thresh, truths_loc, dbox, variance, labels_conf, loc_t, conf_t_label, idx)

        # ここで、
        # loc_tは8732個の要素のうち、Positive DBoxに該当する数だけ有効な数値が入る
        # conf_t_labelは8732個の要素数は変わらず、Positive DBoxはtarget BBoxのクラスラベルが入り、Negative DBoxは背景(0)になる

        # -----
        # 位置の損失：loss_l
        # Smooth L1関数
        # ただし物体を発見したDBoxのオフセットのみを計算する
        # -----

        # 物体を検出したDBox(Positive DBox)を取り出すマスク
        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        # torch.Size([num_batch, 8732]) -> torch.Size([num_batch, 8732, 4])
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Positive DBoxのloc_data(位置補正情報の推論値)と教師データloc_tを取得
        loc_p = loc_data[pos_idx].view(-1, 4)  # Boolean Indexによる抽出後は必ず、1次元配列になるので、形状を変更する
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 物体を発見したPositive DBoxのオフセット情報loc_tの損失(誤差)を計算
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # print("loc_p", loc_p)
        # print("loc_t", loc_t)
        # print("loss_l", loss_l)

        # -----
        # クラス予測の損失: loss_c
        # 交差エントロピー誤差関数
        # 背景クラスが正解のDBoxが圧倒的に多いので、Hard Negative Miningを実施し、
        # 物体発見DBoxと背景クラスDBoxの比が1:3になるようにする。
        # 背景クラスDBoxと予想したもののうち、損失が小さいものはクラス予測の損失から除く
        # -----

        batch_conf = conf_data.view(-1, num_classes)  # (batch_num,8732,21) -> (batch_num*8732,21)
        # print("batch_conf", batch_conf)
        # print("batch_conf size: ", batch_conf.size())

        # クラス予測の損失関数を計算(reduction='none'にして、和を取らずに次元を潰さない)
        # batch_conf size: (batch_num*8732,21), conf_t_label size: (batch_num*8732,)
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none')  # 一旦、すべてのDBoxに対して損失を計算
        # loss_c: (batch_num * 8732,)

        # -----
        # Negative DBoxのうち, Hard Negative Miningで抽出するものを求めるマスクを作成
        # -----

        # 物体を発見したPositive DBoxの損失を0にする
        # (注意) 物体はlabelが1以上.0は背景
        num_pos = pos_mask.long().sum(dim=1,
                                      keepdim=True)  # 各入力データ(画像)毎のPositive Boxの数を取得 (batch_num, 8732) -> (batch_num, 1)
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0  # 物体を発見したDBoxに対応する損失は0にする

        # Hard Negative Miningの実行
        """各DBoxの損失の大きさloss_cの順位であるidx_rankを求める"""
        _, loss_idx = loss_c.sort(dim=1, descending=True)  # 損失に基づいて各DBox(8732)を降順にソート
        _, idx_rank = loss_idx.sort(dim=1)
        # loss_rankは、DBoxの損失を降順にソートした時の元配列のインデックスの並び

        """
        (注釈)
        上２行の実装コードは特殊で直感的でない。
        やりたいことは、各DBoxに対して、損失の大きさが何番目なのかの情報をidx_rankとして高速に取得する。

        DBoxの損失値の大きい方から降順に並べ、DBoxの降順のindexをloss_idxに格納。
        損失の大きさloss_cの順位であるidx_rankを求める。
        ここで、
        降順になった配列indexであるloss_idxを0~8732までの昇順で並べ直すためには、
        何番目のloss_idxのインデックスを取ってきたら良いかを示すのが、idx_rankである。
        例えば、
        idx_rankの要素0番目 = idx_rank[0]を求めるには、loss_idxの値が0の要素、つまり
        loss_idx[?] = 0の?は何番目かを求めることになる。ここで、? = idx_rank[0]である。
        いま、loss_idx[?] = 0の0は、元のloss_cの要素の0番目という意味である。
        つまり、?は、元のloss_cの要素0番目は、降順に並び替えられたloss_idxの何番目ですか
        を求めていることになり、結果、? = idx_rank[0]はloss_cの要素0番目が降順の何番目かを示す。

        e.g
        loss_c                      3.2  5.8  1.3  2.5  4.0
        sorted_loss_c               5.8  4.0  3.2  2.5  1.3
        descending_of_loss_c_index    1    4    0    3    2 (loss_idx)
        sorted_loss_idx               0    1    2    3    4
        ascending_of_loss_idx         2    0    4    3    1 (idx_rank)

        """

        # 背景のDBoxの数num_negを決める。Hard Negative Miningにより、物体を発見したDBoxの数num_posの3倍(self.negpos_ratio)とする。
        # 万が一、DBoxの数を超える場合は、DBoxを上限とする
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)

        # 背景のDBoxの数num_negよりも順位が低い(損失が大きい)DBoxを抽出するマスク
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # -----
        # (終了)
        # -----

        # Negative DBoxのうち、Hard Negative Miningで抽出するものを求めるマスクを作成

        # pos_mask: torch.Size([num_batch, 8732]) -> pos_idx_mask: torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # posとnegだけを取り出してconf_hnmにする。torch.Size([num_pos + num_neg, 21])
        # gtは greater than (>)の略称。これでmaskが1のindexを取り出す。
        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(-1, num_classes)

        # posとnegだけのconf_t_label torch.Size([pos + neg])
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]

        # confidenceの損失関数を計算
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')
        # print("conf_hnm", conf_hnm)
        # print("conf_t_label_num", conf_t_label_hnm)
        # print("loss_c", loss_c)

        # 物体を発見したBBoxの数N(全ミニバッチの合計)で損失を割り算
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
