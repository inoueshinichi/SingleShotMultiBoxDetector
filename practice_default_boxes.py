
import itertools
import math
import torch
import numpy as np
from PIL import Image, ImageDraw

class DBox:

    def __init__(self, img_size: int,
                 dbox_size_steps: list,
                 feature_maps: list,
                 min_dbox_size: list,
                 max_dbox_size,
                 aspect_ratio_list: list):

        self.img_size = img_size
        self.dbox_size_steps = dbox_size_steps
        self.feature_maps = feature_maps
        self.min_dbox_size = min_dbox_size
        self.max_dbox_size = max_dbox_size
        self.aspect_ratio_list = aspect_ratio_list

    def make_dbox_list(self):
        mean = []
        for k, feature_size in enumerate(self.feature_maps):
            for i, j in itertools.product(range(feature_size), repeat=2):

                # DBoxの正規化中心座標
                f_k = self.img_size / self.dbox_size_steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 小さめの正方形
                width = self.min_dbox_size[k] / self.img_size
                height = self.min_dbox_size[k] / self.img_size
                mean.append([cx, cy, width, height])

                # 大きめの正方形
                width = math.sqrt(width * self.max_dbox_size[k] / self.img_size)
                height = math.sqrt(height * self.max_dbox_size[k] / self.img_size)
                mean.append([cx, cy, width, height])

                # アスペクト比を変更した矩形(feature_masのサイズによって種類数は異なる)
                for aspect_ratio in self.aspect_ratio_list[k]:
                    mean.append([cx, cy, width * math.sqrt(aspect_ratio), height / math.sqrt(aspect_ratio)])
                    mean.append([cx, cy, width / math.sqrt(aspect_ratio), height * math.sqrt(aspect_ratio)])

        # meanをTensorに変換し,形状を変更して出力
        output = torch.FloatTensor(mean).view(-1, 4)

        # DBoxが画像の外にはみ出るのを防ぐために大きさを[0, 1]で制限
        output = output.clamp(min=0, max=1)

        return output


if __name__ == "__main__":

    img_size = 300
    dbox_size_steps = [8, 16, 32, 64, 100, 300]
    feature_maps = [38, 19, 10, 5, 3 ,1]
    min_dbox_size = [30, 60, 111, 162, 213, 264]
    max_dbox_size = [45, 99, 153, 207, 261, 315]
    aspect_ratio_list = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    dbox = DBox(img_size=img_size,
                dbox_size_steps=dbox_size_steps,
                feature_maps=feature_maps,
                min_dbox_size=min_dbox_size,
                max_dbox_size=max_dbox_size,
                aspect_ratio_list=aspect_ratio_list)

    dbox_list = dbox.make_dbox_list()
    dboxes = dbox_list.numpy() * img_size
    print("dboxes shape: ", dboxes.shape)
    print("dboxes: ", dboxes)

    bboxes = np.zeros_like(dboxes, dtype=np.float32)
    bboxes[:, :2] = (dboxes[:, :2] - dboxes[:, 2:] / 2)
    bboxes[:, 2:] = (dboxes[:, :2] + dboxes[:, 2:] / 2)
    bboxes = np.clip(bboxes, a_min = 0, a_max = img_size)
    bboxes = bboxes.astype(np.uint32)
    print("bboxes: ", bboxes)

    img = Image.open("cowboy-757575_640.jpg")
    img = img.convert('L').convert('RGB')
    img = img.resize((img_size, img_size))
    # img_array = np.asarray(img)
    # img = Image.fromarray(img_array)
    print(img.format, img.size, img.mode)

    draw = ImageDraw.Draw(img)
    showIdx = []
    sum_box = 0
    for f, ar in zip(feature_maps, aspect_ratio_list):
        sum_box += f**2 * (2 + len(ar) * 2)
        showIdx.append(sum_box)

    # for i in range(bboxes.shape[0]):
    #     draw.rectangle(xy=(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]),
    #                        fill=None, outline=(255, 0, 255))
    #
    #     # for num in showIdx:
    #     #     if i == num - 1:
    #     #         print("bbox_num_idx: ", i)
    #     #         img.show()
    #
    #     if i == 5:
    #         img.show()

    for i in range(1080, 1080+4):
        draw.rectangle(xy=(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]),
                            fill=None, outline=(255, 0, 255))

    img.show()