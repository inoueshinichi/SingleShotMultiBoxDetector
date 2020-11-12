"""Grand Truth Box(正解のBBox)とDBoxの対応づけ
"""

"""
https://github.com/amdegroot/ssd.pytorch
のbox_utils.pyより使用
関数matchを行うファイル
本章の実装はGitHub：amdegroot/ssd.pytorch [4] を参考にしています。
MIT License
Copyright (c) 2017 Max deGroot, Ellis Brown
"""


import torch


def point_form(boxes):
    """
    (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2),
                     dim=1)


def center_size(boxes):
    """
    (xmin, ymin, xmax, ymax) -> (cx, cy, w, h)
    """
    return torch.cat((boxes[:, :2] + boxes[:, 2:]) / 2,
                     boxes[:, 2:] - boxes[:, :2],
                     dim=1)


def intersect(box_a, box_b):
    """
    box_aとbox_bの重なる領域を計算
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Args:
        box_a: [A,4]
        box_b: [B,4]
    Return:
        [A,B]
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Returns:
        ジャカード係数: shape [box_a.size(0), box_b.size(0)]
    """

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter) # [A,B]
    union = area_a + area_b - inter
    return inter / union # [A,B]


def encode(matched, default_boxes, variances):
    """
    Default Box から Ground truth Box への位置補正情報を取得
    Args:
        matched: Ground truth box [xmin, ymin, xmax, ymax] (gtBox_num, 4)
        default_boxes: Default box [cx, cy, w, h] (dBox_num 4)
        variances: 係数

    Returns:
       (*, 4) [[d_cx, d_cy, d_w, d_h], ....]
    """

    # GT Box と Def Boxの中心座標間のxとyの距離
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - default_boxes[:, :2]  # [*,2]

    # 補正
    g_cxcy /= (variances[0] * default_boxes[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / default_boxes[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], dim=1)


def match(jaccard_thresh, gt_boxes, def_boxes, variances, labels):
    """
    ジャッカード係数が最も高いground truths box と default boxを対応させる
    Args:
        jaccard_thresh: ジャッカード係数のしきい値
        gt_boxes: ground truth boxes [[xmin, ymin, xmax, ymax], ...]
        def_boxes: default boxes     [[cx, cy, w, h], ...]
        variances: 位置の補正係数      [0.1, 0.2]
        labels: 画像内にある物体ラベルの種類 [0, 1, 2, ...]
    Returns:
        loc_t: ground truth boxesにマッチしたdefault boxesの位置情報
        conf_t: default boxesとマッチしたground truth boxesの信頼度
    """

    # ジャッカード係数
    overlaps = jaccard(gt_boxes, point_form(def_boxes)) # [A,B], AはGT-Boxの数, BはDef-Boxの数

    # 各GT-Boxに対応するDBoxの中でベストスコアを持つDBoxのジャッカード係数とDBoxのインデックスを取得
    best_dbox_overlap, best_dbox_idx = overlaps.max(dim=1, keepdim=True)   # [画像内のGT-Boxの数, 1]

    # 各DBoxに対応するGT-Boxの中でベストスコアを持つGT-Boxのジャッカード係数とGT-Boxのインデックスを取得
    best_gtbox_overlap, best_gtbox_idx = overlaps.max(dim=0, keepdim=True) # [1, Dboxの数]

    best_gtbox_overlap.squeeze_(0)
    best_gtbox_idx.squeeze_(0)
    best_dbox_overlap.squeeze_(1)
    best_dbox_idx.squeeze_(1)

    # GT-Boxのベストジャッカード係数(0~1)となっている値を2に置き換えている(確実にするためとは？)
    best_gtbox_overlap.index_fill_(dim=0, index=best_dbox_idx, value=2)

    for j in range(best_dbox_idx.size(0)): # GT-Boxの数
        best_gtbox_idx[best_dbox_idx[j]] = j

    matches = gt_boxes[best_gtbox_idx]
    conf = labels[best_gtbox_idx] + 1
    conf[best_gtbox_overlap < jaccard_thresh] = 0  # background
    loc = encode(matches, def_boxes, variances)

    return loc, conf

