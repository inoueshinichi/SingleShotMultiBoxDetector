"""SSDモデルの学習と検証
"""
# 標準
import time

# サードパーティ
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import torch
from torch import optim

# 自作
from make_dataset import make_datapath_list
from convert_xml_annotation_data import Anno_xml2list
from data_argmentation import DataTransform
from make_dataset import VOCDataset
from make_dataloader import make_dataloader_dict
from ssd_model import SSD
from loss_function import MultiBoxLoss


def main():
    """1) 学習データ(画像リスト)(アノテーションリスト)のパスを準備"""
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    """2) アノテーション変換オブジェクト作成"""
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    transform_anno = Anno_xml2list(voc_classes)

    """3) 前処理(data_augumentationを含む)オブジェクト作成"""
    color_mean = (104, 117, 123)  # VOCデータセットの(BGR)平均
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    """4) Pytorch用のデータセットを作成"""
    train_dataset = VOCDataset(train_img_list,
                               train_anno_list,
                               phase='train',
                               transform=transform,
                               transform_anno=transform_anno)
    val_dataset = VOCDataset(val_img_list,
                             val_anno_list,
                             phase='val',
                             transform=transform,
                             transform_anno=transform_anno)

    """5) データローダーの作成"""
    batch_size = 5
    dataloader_dict = make_dataloader_dict(batch_size=batch_size,
                                           train_dataset=train_dataset,
                                           val_dataset=val_dataset,
                                           train_shuffle=True)

    """6) ネットワークモデルの作成"""
    ssd_cfg = {
        'num_classes': 21,
        'input_size': 300,
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],  # 小さい正方形DBoxのピクセルサイズ
        'max_sizes': [60, 111, 162, 213, 264, 315],  # 大きい正方形DBoxのピクセルサイズ
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }
    net = SSD(phase='train', cfg=ssd_cfg)

    # SSDの初期重みを設定
    vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
    net.vgg.load_state_dict(vgg_weights)  # 変更したVGG16の下位2層のconv2dを含めてVGG16を初期化

    # print("vgg_weights", vgg_weights)

    # SSDのその他のネットワーク(extra,loc,conf)はHeの初期値
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.extra.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    # GPUが使えるか確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("使用デバイス:", device)
    print("ネットワーク設定完了：学習済み重みをロードしました。")

    """7) 損失関数と最適化手法の設定"""
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    """8) 学習と検証の実施"""
    # ネットワークがある程度固定であれば、高速化させる(再現性はない)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    net = net.to(device)
    print("net", net)
    # criterion = criterion.to(device)
    # print("criterion", criterion)

    # イテレーションカウンタをセット
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # epochs
    num_epochs = 20

    # epoch loop
    for epoch in range(num_epochs + 1):
        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("----------")
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("----------")

        # epoch毎の訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print(' (train) ')
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()
                    print("----------")
                    print(' (val) ')
                else:
                    continue  # 検証は10回に1回行う

            # dataloaderからminibatchずつ取り出すループ
            for images, targets in dataloader_dict[phase]:
                # GPUが使えるならGPUにデータを転送
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化
                optimizer.zero_grad()

                # print("--> iter {}".format(phase))

                # 順伝播計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 訓練時は逆誤差伝播
                    if phase == 'train':
                        loss.backward()

                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも2.0に留める
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step()  # パラメータ更新

                        if iteration % 10 == 0:  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print("イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec.".format(
                                iteration, loss.item(), duration
                            ))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item()

        # epochのphase毎のlossと正解率
        t_epoch_finish = time.time()
        print("----------")
        print("epoch {} || Epoch_TRAIN_Loss: {:.4f} || Epoch_VAL_Loss: {:.4f}".format(
            epoch + 1, epoch_train_loss, epoch_val_loss
        ))

        print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch + 1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # ネットワークを保存
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), "weights/ssd300_" + str(epoch + 1) + ".pth")


if __name__ == "__main__":
    main()