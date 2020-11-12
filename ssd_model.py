"""SSD用ネットワークモデル
"""
import torch
from torch import nn

"""
    VGG   -> VGG16モデル. source1とsource2を出力
    Extra -> source2からsource3, source4, source5, source6を出力
    Loc   -> source1 ~ source6 から8732個のDBoxのオフセット情報を出力  : torch.Size([1, 8732, 4])
    Conf  -> source1 ~ source6 から8732個のDBoxの確信度情報を出力     : torch.Size([1, 8732, 4])  
"""



"""改良VGGモジュール
    inputs: img_data[n x 300 x 300 x 3]

    outputs: conv4_3[n x ○ x ○ x 512]
             source2[n x 19 x 19 x 1024]

    net:
        1) 300 x 300 x 3
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU : 300 x 300 x 64
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU : 300 x 300 x 64
           MaxPool2d(kernel_size=2, s=2, p=0) : 150 x 150 x 128

        2) 150 x 150 x 128
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU : 150 x 150 x 128
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU : 150 x 150 x 128
           MaxPool2d(kernel_size=2, s=2, p=0) : 75 x 75 x 256

        3) 75 x 75 x 256
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  75 x 75 x 256
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  75 x 75 x 256
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  75 x 75 x 256
           Ceiling_MaxPool2d(kernel_size=2, s=2, p=0) : 38 x 38 x 512

        4) 38 x 38 x 512
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  38 x 38 x 512
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  38 x 38 x 512
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  38 x 38 x 512 ----> *[L2Norm]* -----> source1
           MaxPool2d(kernel_size=2, s=2, p=0) : 19 x 19 x 512

        5) 19 x 19 x 512
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  19 x 19 x 512
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  19 x 19 x 512
           Conv2d(kernel_size=3, s=1, p=1, d=1, zero_padding) + ReLU :  19 x 19 x 512
           MaxPool2d(kernel_size=3, s=1, p=1) : 19 x 19 x 1024

        6) 19 x 19 x 1024
           Conv2d(kernel_size=3, s=1, p=6, d=6, zero_padding) + ReLU :  19 x 19 x 1024 ※ Dilated Convolution
           Conv2d(kernel_size=3, s=1, p=0, d=1, zero_padding) + ReLU :  19 x 19 x 1024 ※ Dilated Convolution ----> source2
"""
def make_vgg():
    # 34層にわたるvggモジュールを作成
    layers = []
    in_channels = 3

    # vggモジュールで使用する畳み込み層やMaxPooling層のチャンネル数
    cfg = [64, 64, 'M',
           128, 128, 'M',
           256, 256, 256, 'MC',
           512, 512, 512,'M',
           512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [ nn.MaxPool2d(kernel_size=2, stride=2) ]
        elif v == 'MC':
            # ceilは出力サイズを、計算結果(float)に対して、切り上げで整数にするモード
            # デフォルト(floor)では出力サイズを計算結果(float)に対して、切り下げで整数にするモード
            layers += [ nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) ]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [ conv2d, nn.ReLU(inplace=True) ]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)


"""Extraモジュール
    inputs : source2[n x 19 x 19 x 1024]

    outputs : 
            source3[n x 10 x 10 x 512]
            source4[n x 5 x 5 x 256]
            source5[n x 3 x 3 x 512]
            source6[n x 1 x 1 x 512]

    net: 
            1) n x 19 x 19 x 1024
              Conv2d(kernel_size=1, s=1, p=0, d=1, zero_padding) : 19 x 19 x 256
              Conv2d(kernel_size=3, s=2, p=1, d=1, zero_padding) : 10 x 10 x 512 ----> source3

            2) n x 10 x 10 x 512
              Conv2d(kernel_size=1, s=1, p=0, d=1, zero_padding) : 10 x 10 x 512
              Conv2d(kernel_size=3, s=2, p=1, d=1, zero_padding) : 5 x 5 x 256 ----> source4

            3) n x 5 x 5 x 256
              Conv2d(kernel_size=1, s=1, p=0, d=1, zero_padding) : 5 x 5 x 256
              Conv2d(kernel_size=3, s=1, p=0, d=1, zero_padding) : 3 x 3 x 512 ----> source5

            4) n x 3 x 3 x 512
              Conv2d(kernel_size=1, s=1, p=0, d=1, zero_padding) : 3 x 3 x 512
              Conv2d(kernel_size=3, s=1, p=0, d=1, zero_padding) : 1 x 1 x 512 ----> source6
"""
def make_extra():
    # 8層に渡るextraモジュールを作成
    layers = []
    in_channels = 1024 # vggの出力のチャネル数

    # extraモジュールの畳み込み層のチャネル数を設定する
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=3)]

    return nn.ModuleList(layers)


"""Locモジュール
    inputs :
            source1[n x 38 x 38 x 512]
            source2[n x 19 x 19 x 1024]
            source3[n x 10 x 10 x 512]
            source4[n x 5 x 5 x 256]
            source5[n x 3 x 3 x 512]
            source6[n x 1 x 1 x 512]

    outputs: torch.size([1, 8732, 4])

    net:
        source1 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x 38 x 38 x 4 x 4] (小正方形, 大正方形, 縦長長方形, 横長長方形) x (Δcx, Δcy, Δwidth, Δheight)
        source2 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 6 x 4] (小正方形, 大正方形, 縦長長方形, 横長長方形, 超縦長長方形, 超横長長方形) x (Δcx, Δcy, Δwidth, Δheight)
        source3 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 6 x 4] (小正方形, 大正方形, 縦長長方形, 横長長方形, 超縦長長方形, 超横長長方形) x (Δcx, Δcy, Δwidth, Δheight)
        source4 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 6 x 4] (小正方形, 大正方形, 縦長長方形, 横長長方形, 超縦長長方形, 超横長長方形) x (Δcx, Δcy, Δwidth, Δheight)
        source5 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 4 x 4] (小正方形, 大正方形, 縦長長方形, 横長長方形) x (Δcx, Δcy, Δwidth, Δheight)
        source6 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 4 x 4] (小正方形, 大正方形, 縦長長方形, 横長長方形) x (Δcx, Δcy, Δwidth, Δheight)
"""

"""Confモジュール
    inputs :
            source1[n x 38 x 38 x 512]
            source2[n x 19 x 19 x 1024]
            source3[n x 10 x 10 x 512]
            source4[n x 5 x 5 x 256]
            source5[n x 3 x 3 x 256]
            source6[n x 1 x 1 x 256]

    outputs: torch.size([1, 8732, 21])

    net:
        source1 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 4 x 21] (小正方形, 大正方形, 縦長長方形, 横長長方形) x 21種類のクラスラベル
        source2 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 6 x 21] (小正方形, 大正方形, 縦長長方形, 横長長方形, 超縦長長方形, 超横長長方形) x 21種類のクラスラベル
        source3 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 6 x 21] (小正方形, 大正方形, 縦長長方形, 横長長方形, 超縦長長方形, 超横長長方形) x 21種類のクラスラベル
        source4 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 6 x 21] (小正方形, 大正方形, 縦長長方形, 横長長方形, 超縦長長方形, 超横長長方形) x 21種類のクラスラベル
        source5 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 4 x 21] (小正方形, 大正方形, 縦長長方形, 横長長方形) x 21種類のクラスラベル
        source6 ----> Conv2d(kernel_size=3, s=1, p=1, zero_padding) : [n x ○ x ○ x 4 x 21] (小正方形, 大正方形, 縦長長方形, 横長長方形) x 21種類のクラスラベル
"""


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    # デフォルトボックスのオフセットを出力するloc_layers
    # デフォルトボックスに対する各クラスの信頼度confidenceを出力するconf_layers

    loc_layers = []
    conf_layers = []

    # VGGの22層目, conv4_3(source1)に対する畳込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]

    # VGGの最終層(source2)に対する畳込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]

    # extraの(source3)に対する畳み込み
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # extraの(source4)に対する畳込み
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # extraの(source5)に対する畳込み
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # extraの(source6)に対する畳込み
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


"""L2ノルム層
"""
class L2Norm(nn.Module):
    # ConvC4_3からの出力をscale=20のL2normで正規化する

    def __init__(self, input_channels=512, scale=20):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # 係数weightsをscaleで初期化する
        self.reset_parameters()  # パラメータの初期化
        self.eps = 1e-10

    def reset_parameters(self):
        nn.init.constant_(self.weights, self.scale)  # 全てのweightをscale=20で初期化

    def forward(self, x):
        '''
        38x38の特徴量に対して、512チャネルに渡って2乗和をのルートを求めた38x38個の値を使用し、
        各特徴量を正規化してから係数を掛け算する層
        '''

        # normの計算
        # normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # 係数の次元を調整
        # self.weightsのサイズはtorch.Size([512])なので、
        # torch.Size([batch_num, 512, 38, 38])まで変形する
        weights = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        # 正規化
        out = x * weights

        # print("out size: ", out.size())

        return out



"""SSDモデルクラスを実装
"""
import torch.nn.functional as F

from inference import Detect
from default_boxs import DBox


class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super().__init__()

        self.phase = phase  # train or interfaceを指定
        self.num_classes = cfg['num_classes']  # クラス数=21

        # SSDネットワーク
        self.vgg = make_vgg()
        self.extra = make_extra()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg['num_classes'], cfg['bbox_aspect_num'])

        # DBox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()
        print("self.dbox_list's length: ", len(self.dbox_list))

        # 推論次はクラス"Detect"を用意する
        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        """
            input: (batch_num, 1)の画像

            1. source1にL2Normを適用
            2. source2を計算
            3. source2からsource3~source6を計算
            4. source2~source6からlocを計算
            5. source2~source6からconfを計算
            6. loc, conf, dboxをDetectに通過させ、conf>0.01, IoU>0.45を満たすBBoxを求める

            output :
                BBox : (batch_num, 21, 200, 5)
        """

        # print("x size: ", x.size())
        # print("input x in forward", x)

        sources = list()  # source1~6を格納
        loc = list()  # locを格納
        conf = list()  # confを格納

        # 1) vggのconv4_3まで計算
        for k in range(23):
            x = self.vgg[k](x)
            # if k == 0:
            #     print("No.{0} x \n{1}".format(k, x))

        # 2) conv4_3の出力をL2Normに入力して,source1を作成
        source1 = self.L2Norm(x)
        sources.append(source1)

        # 3) vggを最後まで計算してsource2を作成
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)  # source2

        # 4) extraのconvとReLUを計算
        # source3~source6をsourcesに追加
        for k, v in enumerate(self.extra):  # k : 0 ~ 7
            # 引数なしかinplace=Falseとすると、入力したtensorとは別のtensorが返ってくる。
            # inplace=Trueとすると、入力したtensorをそのまま書き換えて返す。
            # 直接書き換えた方がメモリー使用を少なくできる
            x = F.relu(v(x), inplace=True)
            # print("x in extra", x)

            # 偶数番目はsource*なのでsourcesに追加
            if k % 2 == 1:
                # print("x size: ", x.size())
                sources.append(x)

        # print("sources len: ", len(sources))

        # source1~source6にそれぞれ対応する畳み込みを1回ずつ適用する
        # sources, self.loc, self.conf共に要素数6のリスト
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permuteは要素の順番を入れ替える
            # l(x)とc(x)で畳み込みを実行
            # l(x)とc(x)の出力サイズは[batch_num, 4 * アスペクト比の種類数, featureマップの高さ, featureマップの幅]
            # sourceによってアスペクト比の種類が異なる([2] or [2, 3])ので4 * アスペクト比の種類数を4次元目に移動させる
            # permuteで要素の順番を入れ替える。
            # [minibatch_size, featuremap高さ, featuremap幅, 4 * アスペクト比の種類]
            # (注釈)
            # torch.contiguos()はメモリ上で要素を連続的に配置し直す命令。
            # 後でview関数を使用するが、view関数を行うためには、対象の変数がメモリ上で連続配置されている必要があるから。
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # locとconfの変形(1)
        # loc : torch.Size([batch_num, 34928])
        # conf: torch.Size([batch_num, 183372])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)
        # print("loc size: ", loc.size())
        # print("conf size: ", conf.size())

        # locとconfの変形(2)
        # loc : torch.Size([batch_num, 8732, 4])
        # conf: torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        # print("loc in forward of SSD Model", loc)
        # print("conf in forward of SSD Model", conf)

        # 最後に出力
        output = (loc, conf, self.dbox_list)
        # print("loc size: ", loc.size())
        # print("conf size: ", conf.size())
        # print("dbox_list len: ", len(self.dbox_list))

        # 推論時と学習時で挙動を変える
        if self.phase == 'inference':
            # Detectクラスのforwarを実行
            # 戻り値のサイズはtorch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])
        else:
            # 学習時
            # 戻り値は(loc, conf, dbox_list)のタプル
            return output


if __name__ == "__main__":

    # VGGモジュール
    vgg_test = make_vgg()
    print(vgg_test)

    # EXTRAモジュール
    extra_test = make_extra()
    print(extra_test)

    # LOC&CONFモジュール
    loc_test, conf_test = make_loc_conf()
    print(loc_test)
    print(conf_test)

    # SSD
    SSD300_cfg = {
        'num_classes': 21,  # 背景クラスを含めた合計クラス数
        'input_size' : 300, # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4,],    # 出力するDBoxのアスペクト比の種類
        'feature_maps': [38, 19, 10, 5, 3, 1],     # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300] ,       # DBoxのピクセルサイズ
        'min_sizes': [30, 60, 111, 162, 213, 264], # 小さい正方形のDBoxのピクセルサイズ
        'max_sizes': [60, 111, 162, 213, 264, 315], # 大きい正方形のDBoxのピクセルサイズ
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]] # アスペクト比の構成?
    }
    ssd_test = SSD(phase="train", cfg=SSD300_cfg)
    print(ssd_test)
