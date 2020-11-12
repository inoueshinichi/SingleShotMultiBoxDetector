"""SSDクラスのfoward処理を実装する
"""
import torch


"""SSDの推論時には、順伝搬の最後にクラスDetectを用いる。
   Detectクラスの中で使用する関数decodeと関数nm_suppressionを実装

　　関数decode:
        DBox=(cx_d, cy_d, w_d, h_d)とSSDモデルから求めたオフセット情報loc=(Δcx, Δcy, Δw, Δh)を使用し、
        BBoxの座標情報を作成する.

        BBoxの情報は、
        cx = cx_d * (1 + 0.1 * Δcx)
        cy = cy_d * (1 + 0.1 * Δcy)
        w  = w_d * exp(0.2 * Δw)
        h  = h_d * exp(0.2 * Δh)

        さらに、表示形式を(cx, cy, w, h)-> (xmin, ymin, xmax, ymax)に変換する.
"""
def decode(loc, dbox_list):
    """オフセット情報を使い、DBoxをBBoxに変換

    Arguments:
        loc {[8732, 4]} -- [SSDモデルで推論するオフセット情報]
        dbox_list {[8732, 4]} -- [DBoxの情報]

    Return:
        bboxes : [xmin, ymin, xmax, ymax]
    """

    # DBoxは[cx, cy, width, height]
    # locは[Δcx, Δcy, Δw, Δh]
    """変換式
        cx = cx_d * (1 + 0.1 * Δcx)
        cy = cy_d * (1 + 0.1 * Δcy)
        w  = w_d * exp(0.2 * Δw)
        h  = h_d * exp(0.2 * Δh)
    """
    centers = dbox_list[:, :2] + dbox_list[:, :2] * 0.1 * loc[:, :2]
    sizes = dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)
    bboxes = torch.cat((centers, sizes), dim=1)

    # 表示形式を(cx, cy, w, h)-> (xmin, ymin, xmax, ymax)に変換する
    bboxes[:, :2] -= bboxes[:, 2:] / 2  # [xmin, ymin] = [cx - w/2, cy - h/2]
    bboxes[:, 2:] += bboxes[:, :2]  # [xmax, ymax] = [xmin + w, ymin + h]

    return bboxes


# Non-Maximum-Suppression処理
# 同じ物体クラスを指し示す複数のBBoxがある場合に、
# 閾値overlap = 0.45以上のBBoxは冗長なBBoxとして排除して、
# 残ったBBoxの中で最も確信度Confが高いBBoxを残す
def NonMaximum_Suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    # bboxes : [確信度0.01を超えたBboxの数, 4] loc情報
    # scores : [確信度0.01を超えたBBoxの数]    conf情報
    Args:
        boxes:
        scores:
        overlap:
        top_k:

    Returns:
        keep: list, confの降順にnmsを通過したindexが格納される
        count: int, nmsを通過したBBoxの数
    """

    # returnの雛形を作成
    count = 0
    # keep = scores.new_tensor(scores).zero_().long()
    keep = scores.clone().zero_().long()
    # print("keep.size()", keep.size())
    # print("keep", keep)
    # print("scores", scores)

    # keep : torch.Size([確信度を超えたBBoxの数]) 要素は全部0

    # BBoxの領域を計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # # BBoxの被り度合いIoUの計算に使用する変数
    # tmp_x1 = boxes.new()
    # tmp_y1 = boxes.new()
    # tmp_x2 = boxes.new()
    # tmp_y2 = boxes.new()
    # tmp_w = boxes.new()
    # tmp_h = boxes.new()
    # 事前にメモリを確保する必要ない by inoue shinichi 2020.06.06

    # scoresを昇順ソート
    sorted_scores, idx = scores.sort(dim=0)  # 0次元でソート

    # 上位top_k個(200個)のBBoxのindexを取り出す(200個存在しない場合もある)
    idx_top_k = idx[-top_k:]

    # idx_top_kの要素数が0出ない限りループ
    while idx_top_k.numel() > 0:
        i = idx_top_k[-1]  # 現在のconf最大のindexをiにセット

        # keepにconf最大のindexをセット
        keep[count] = i
        count += 1

        # 最後のBBoxになった場合は、break
        if idx_top_k.size(0) == 1:
            break

        # idx_top_kの最後の要素を一つ減らす
        idx_top_k = idx_top_k[:-1]

        """keepに格納したBBoxと被りが大きいBBoxを抽出して消去
        """
        # 1つ要素を減らしたidx_top_kまでのBBoxを,outに指定した変数として作成
        tmp_x1 = torch.index_select(x1, dim=0, index=idx_top_k)  # , out=tmp_x1)
        tmp_y1 = torch.index_select(y1, dim=0, index=idx_top_k)  # , out=tmp_y1)
        tmp_x2 = torch.index_select(x2, dim=0, index=idx_top_k)  # , out=tmp_x2)
        tmp_y2 = torch.index_select(y2, dim=0, index=idx_top_k)  # , out=tmp_y2)

        # 全てのBBoxに対して、現在のBBox=indexがiとかぶっている値までに設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i].item())
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i].item())
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i].item())
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i].item())

        # print("tmp_x2", tmp_x2)
        # print("tmp_w", tmp_w)

        # # wとhのテンソルサイズを1つ減らしたものにする
        # tmp_w.resize_as_(tmp_x2) # tmp_x2と同じ形状にリサイズする。ただし、tmp_x2の要素数が小さいとtmp_wの一部を削除する
        # tmp_h.resize_as_(tmp_y2)
        # 事前にメモリを確保していないので必要ない by inoue shinichi 2020.06.06

        # clampした状態でのBBoxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1  # このタイミングで初めてtmp_wのメモリが生成される
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さが負に成っているものは0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clampされた状態での面積(かぶっている領域)
        inter = tmp_w * tmp_h

        # IoU = intersect / Union
        rem_areas = torch.index_select(area, dim=0, index=idx_top_k)  # 各BBoxの元の面積
        union = area[i] + (rem_areas - inter)  # union
        IoU = inter / union

        # IoUがoverlapより小さいidx_top_kのみを残す
        idx_top_k = idx_top_k[IoU.le(overlap)]  # leは Less than or Equal to

    return keep, count


"""クラスDetect
    output : 
        (batch_num, 21, 200, 5)
            batch_num : バッチサイズ
            21        : クラスラベルの数
            200       : 信頼度上位200個のBBox
            5         : (conf, xmin, ymin, width, height)

    input :
        ※ 8732はスケールを様々に変えたアンカーボックスの数
        loc  : 各アンカーボックスのオフセット情報    (batch_num, 8732, 4)
        conf : 各アンカーボックスに対するクラスラベル (batch_num, 8732, 21)   
        dbox : 各アンカーボックスの位置情報         (8732, 4)
"""
# class Detect(torch.autograd.Function):
class Detect(torch.nn.Module):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        super().__init__()

        # 確信度confを正規化する
        self.softmax = torch.nn.Softmax(dim=-1)

        # conf_thresh
        self.conf_thresh = conf_thresh

        # non-maximum-suppressionで各推定BBoxの上位top_k個を使う
        self.top_k = top_k

        # IoUの閾値nms_thresh
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        """
            クラスラベル数21個に属する上位top_kに入る確信度を持つオフセットを施した
            各BBox(decoded_boxes)にnon-maximum-suppressionを適用してBBoxを絞り、
            1枚の画像の中で必要なBBoxを得る

            input :
                ※ 8732はスケールを様々に変えたアンカーボックスの数
                loc  : 各アンカーボックスのオフセット情報    (batch_num, 8732, 4)
                conf : 各アンカーボックスに対するクラスラベル (batch_num, 8732, 21)
                dbox : 各アンカーボックスの位置情報         (8732, 4)

            output:
                torch.Size([batch_num, 21, 200, 5])
        """

        num_batch = loc_data.size(0)  # バッチサイズ
        num_classes = conf_data.size(2)  # クラス数

        # confは正規化
        conf_data = self.softmax(conf_data)

        # 出力の型
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # conf_data: (batch_num, 8732, num_classes) -> (batch_num, num_classes, 8732)
        conf_pred = conf_data.transpose(1, 2)

        for i in range(num_batch):

            # 1) BBox(8732, 4)を求める
            decoded_boxes = decode(loc_data[i], dbox_list)

            # 2) confのコピー(21, 8732)
            conf_scores = conf_pred[i].clone()

            # 3) 画像クラス毎のループ(背景クラスのindexである0は計算せず、index=1から)
            for cl in range(1, num_classes):

                # 4) conf>0.01に該当するマスク(c_mask)を作成[True, False, False, ...]
                # torch.Size([8732])
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # > 0.01

                # 5) 該当するconfを抽出
                scores = conf_scores[cl][c_mask]
                # print("type(scores)", type(scores))
                # print("scores", scores)

                """該当するBBoxを抽出
                """

                # conf_thresh閾値を超えたconfがない場合(socres=[])
                # 何もしない
                if scores.numel() == 0:
                    continue

                # c_maskをdecoded_boxesに適用できるように次元を調整
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                # 該当するBBoxを抽出
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]で1次元になってしまうので、viewで(閾値を超えたBBox数, 4)サイズに変形

                # Non-Maximum-Suppressionを実行
                # print("boxes", boxes)
                idx, count = NonMaximum_Suppression(boxes, scores, self.nms_thresh, self.top_k)
                # idx   : confの降順にNon-Maximum Suppressionを通過したindexが格納されている
                # count : Non-Maximum Suppressionを通過したBBoxの数

                # outputにNon-Maximum Suppressionを通過した結果を格納(2軸の要素数200以下で値が入る)
                output[i, cl, :count] = torch.cat(
                    (scores[idx[:count]].unsqueeze(1), boxes[idx[:count]]),
                    dim=1)
                # (確信度, xmin, ymin, xmax, ymax)

        return output  # (batch_num, classes_num, top_k, 5)
