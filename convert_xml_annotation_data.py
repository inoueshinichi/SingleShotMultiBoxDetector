"""xml形式のアノテーションデータをリストに変換
"""

import xml.etree.ElementTree as ET
import numpy as np
import cv2


# xmlをlistに変換するクラス
class Anno_xml2list(object):

    def __init__(self, classes):
        """
        params:
            classes : list
                VOCのクラス名を格納したリスト
        """
        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        1枚の画像に対する「xml形式のアノテーションデータ」を画像サイズで規格化してからリスト形式に変換する

        params:
            xml_path: str
                xmlファイルへのパス
            width : int
                画像の横幅
            height : int
                画像の高さ

        return:
            ret : [[xmin, ymin, xmax, ymax, label_id], ....]
        """

        # 画像内の全ての物体のアノテーションをこのリストに格納します
        ret = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体(object)の数だけループする
        for obj in xml.iter('object'):

            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 1つの物体に対するアノテーションを格納するリスト(xmin, ymin, xmax, ymax, index)
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 物体名
            bbox = obj.find('bndbox')  # バウンディングボックス

            # アノテーションのxmin, ymin, xmax, ymaxを取得し、0~1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOCは原点が(1,1)なので、1を引いて(0,0)を原点にする
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅と高さで規格化
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # retに[xmin, ymin, xmax, ymax, label_idx]を足す
            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


if __name__ == "__main__":

    from make_dataset import make_datapath_list
    # ファイルパスのリストを作成
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

    transform_anno = Anno_xml2list(voc_classes)

    # 画像の読み込み OpenCVを使用
    ind = 1
    image_file_path = val_img_list[ind]
    img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
    height, width, channels = img.shape  # 画像のサイズを取得

    print(image_file_path)

    # アノテーションをリストで表示
    transformed_annotation = transform_anno(val_anno_list[ind], width, height)
    print("voc-class's image[{0}]-annotation: \n{1}".format(ind, transformed_annotation))