"""Pytorch用のDataLoaderを作成
"""
import torch
import torch.utils.data as data


def make_dataloader_dict(batch_size, train_dataset=None, val_dataset=None, train_shuffle=True):

    # PytorchのDataLoaderに使用するcollate_fnのオーバーライド
    def object_detect_collate_fn(batch):
        """
        Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
        画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
        この変化に対応したDataLoaderを作成するために、
        カスタイマイズした、collate_fnを作成します。
        collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
        ミニバッチ分の画像が並んでいるリスト変数batchに、
        ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
        """

        # batch -> img and true_boxes_labels
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])  # sample[0]は画像
            targets.append(torch.FloatTensor(sample[1]))  # sample[1]はtrue_boxes_labels

        # imgsはミニバッチサイズのリストになっています
        # リストの要素はtorch.Size([3, 300, 300])です。
        # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
        imgs = torch.stack(imgs, dim=0)

        # targetsはアノテーションデータの正解であるgtのリストです。
        # リストのサイズはミニバッチサイズです。
        # リストtargetsの要素は [n, 5] となっています。
        # nは画像ごとに異なり、画像内にある物体の数となります。
        # 5は [xmin, ymin, xmax, ymax, class_index] です
        return imgs, targets
    # end collate_fn

    # training
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=train_shuffle,
                                       collate_fn=object_detect_collate_fn)

    # valdation
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,  # validataionデータはシャッフルしない
                                     collate_fn=object_detect_collate_fn)

    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloader_dict


if __name__ == "__main__":

    # 1) 学習データ
    from make_dataset import make_datapath_list

    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    # 2) アノテーション変換
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    from convert_xml_annotation_data import Anno_xml2list

    transform_anno = Anno_xml2list(voc_classes)

    # 3) 前処理(data_augumentationを含む)
    color_mean = (104, 117, 123)  # VOCデータセットの(BGR)平均
    input_size = 300
    from data_argmentation import DataTransform

    transform = DataTransform(input_size, color_mean)

    # 4) Pytorch用のデータセットを作成
    from make_dataset import VOCDataset

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

    # 5) データローダーの作成
    batch_size = 5
    dataloader_dict = make_dataloader_dict(batch_size=batch_size,
                                           train_dataset=train_dataset,
                                           val_dataset=val_dataset,
                                           train_shuffle=True)

    # 6) イテレータの設定
    batch_iterator = iter(dataloader_dict["val"])  # イテレータに変換
    imgs, targets = next(batch_iterator)  # 1番目の要素を取り出す
    print(imgs.size())  # torch.Size([4, 3, 300, 300])
    print(len(targets))

    # 7) ミニバッチサイズのリスト
    for i in range(batch_size):
        print("targets[{}]: {}".format(i, targets[i]))

    # 8) データ数
    print("train_dataset_len: {}".format(train_dataset.__len__()))
    print("val_dataset_len: {}".format(val_dataset.__len__()))
