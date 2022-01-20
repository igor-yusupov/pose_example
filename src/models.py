# from collections import OrderedDict

# import timm
import torch
import torch.nn as nn
import yaml


class CocoPoseNet(nn.Module):
    insize = 368

    def __init__(self, path=None):
        super(CocoPoseNet, self).__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)
        if path:
            self.base.vgg_base.load_state_dict(torch.load(path))

    def forward(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(torch.cat([h1, h2, feature_map], dim=1))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps


class VGG_Base(nn.Module):
    def __init__(self):
        super(VGG_Base, self).__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1,
            padding=1)
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128,  kernel_size=3, stride=1,
            padding=1)
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256,  kernel_size=3, stride=1,
            padding=1)
        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256,  kernel_size=3, stride=1,
            padding=1)
        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256,  kernel_size=3, stride=1,
            padding=1)
        self.conv3_4 = nn.Conv2d(
            in_channels=256, out_channels=256,  kernel_size=3, stride=1,
            padding=1)
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512,  kernel_size=3, stride=1,
            padding=1)
        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512,  kernel_size=3, stride=1,
            padding=1)
        self.relu = nn.ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        return x


class Base_model(nn.Module):
    def __init__(self):
        super(Base_model, self).__init__()
        self.vgg_base = VGG_Base()
        self.conv4_3_CPM = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=3, stride=1,
            padding=1)
        self.conv4_4_CPM = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x


class Stage_1(nn.Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1_CPM_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv2_CPM_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv3_CPM_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv4_CPM_L1 = nn.Conv2d(
            in_channels=128, out_channels=512, kernel_size=1, stride=1,
            padding=0)
        self.conv5_CPM_L1 = nn.Conv2d(
            in_channels=512, out_channels=38, kernel_size=1, stride=1,
            padding=0)
        self.conv1_CPM_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv2_CPM_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv3_CPM_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=1)
        self.conv4_CPM_L2 = nn.Conv2d(
            in_channels=128, out_channels=512, kernel_size=1, stride=1,
            padding=0)
        self.conv5_CPM_L2 = nn.Conv2d(
            in_channels=512, out_channels=19, kernel_size=1, stride=1,
            padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x))  # branch1
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv1_CPM_L2(x))  # branch2
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2


class Stage_x(nn.Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = nn.Conv2d(
            in_channels=185, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv2_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv3_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv4_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv5_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv6_L1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.conv7_L1 = nn.Conv2d(
            in_channels=128, out_channels=38, kernel_size=1, stride=1,
            padding=0)
        self.conv1_L2 = nn.Conv2d(
            in_channels=185, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv2_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv3_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv4_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv5_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=7, stride=1,
            padding=3)
        self.conv6_L2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=1, stride=1,
            padding=0)
        self.conv7_L2 = nn.Conv2d(
            in_channels=128, out_channels=19, kernel_size=1, stride=1,
            padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.conv1_L1(x))  # branch1
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        h2 = self.relu(self.conv1_L2(x))  # branch2
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)
        return h1, h2

# class OpenPose(nn.Module):
#     def __init__(self, conf):
#         super(OpenPose, self).__init__()
#         self.conf = conf
#         self.pose_model = bodypose_model()
#
#         self.valid_loss_dict = conf['criterion']['loss_weights']
#
#     def forward(self, data_dict, phase='train'):
#         paf_out, heatmap_out = self.pose_model(data_dict['img'])
#
#         out_dict = dict(paf=paf_out[-1], heatmap=heatmap_out[-1])
#         if phase == 'test':
#             # return {'paf': torch.mean(paf_out, axis=0),
#             #         'heatmap': torch.mean(heatmap_out, axis=0)}
#             return out_dict
#
#         loss_dict = dict()
#         for i in range(len(paf_out)):
#             if 'paf_loss{}'.format(i) in self.valid_loss_dict:
#                 loss_dict['paf_loss{}'.format(i)] = dict(
#                     params=[paf_out[i]*data_dict['maskmap'],
#                             data_dict['vecmap']],
#                     weight=torch.cuda.FloatTensor(
#                         [self.valid_loss_dict['paf_loss{}'.format(i)]])
#                 )
#
#         for i in range(len(heatmap_out)):
#             if 'heatmap_loss{}'.format(i) in self.valid_loss_dict:
#                 loss_dict['heatmap_loss{}'.format(i)] = dict(
#                     params=[heatmap_out[i]*data_dict['maskmap'],
#                             data_dict['heatmap']],
#                     weight=torch.cuda.FloatTensor(
#                         [self.valid_loss_dict['heatmap_loss{}'.format(i)]])
#                 )
#
#         return out_dict, loss_dict
#
#
# class bodypose_model(nn.Module):
#     def __init__(self):
#         super(bodypose_model, self).__init__()
#
#         # these layers have no relu layer
#         no_relu_layers = [
#             'conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',
#             'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',
#             'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',
#             'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
#         blocks = {}
#         block0 = OrderedDict([
#                       ('conv1_1', [3, 64, 3, 1, 1]),
#                       ('conv1_2', [64, 64, 3, 1, 1]),
#                       ('pool1_stage1', [2, 2, 0]),
#                       ('conv2_1', [64, 128, 3, 1, 1]),
#                       ('conv2_2', [128, 128, 3, 1, 1]),
#                       ('pool2_stage1', [2, 2, 0]),
#                       ('conv3_1', [128, 256, 3, 1, 1]),
#                       ('conv3_2', [256, 256, 3, 1, 1]),
#                       ('conv3_3', [256, 256, 3, 1, 1]),
#                       ('conv3_4', [256, 256, 3, 1, 1]),
#                       ('pool3_stage1', [2, 2, 0]),
#                       ('conv4_1', [256, 512, 3, 1, 1]),
#                       ('conv4_2', [512, 512, 3, 1, 1]),
#                       ('conv4_3_CPM', [512, 256, 3, 1, 1]),
#                       ('conv4_4_CPM', [256, 128, 3, 1, 1])
#                   ])
#         block0_0 = OrderedDict([
#             ('conv4_3_CPM', [512, 256, 3, 1, 1]),
#             ('conv4_4_CPM', [256, 128, 3, 1, 1])
#         ])
#
#         # Stage 1
#         block1_1 = OrderedDict([
#                         ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
#                         ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
#                         ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
#                         ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
#                         ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
#                     ])
#
#         block1_2 = OrderedDict([
#                         ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
#                         ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
#                         ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
#                         ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
#                         ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
#                     ])
#         blocks['block1_1'] = block1_1
#         blocks['block1_2'] = block1_2
#
#         # self.model0 = self.make_layers(block0, no_relu_layers)
#         self.model0_0 = self.make_layers(block0_0, no_relu_layers)
#         self.backbone = timm.create_model(
#             model_name='vgg19',
#             pretrained=True)
#         self.model0 = list(self.backbone.children())[0][:23]
#         print(self.model0)
#
#         # Stages 2 - 6
#         for i in range(2, 7):
#             blocks['block%d_1' % i] = OrderedDict([
#                     ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
#                     ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
#                     ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
#                 ])
#
#             blocks['block%d_2' % i] = OrderedDict([
#                     ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
#                     ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
#                     ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
#                     ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
#                 ])
#
#         for k in blocks.keys():
#             blocks[k] = self.make_layers(blocks[k], no_relu_layers)
#
#         self.model1_1 = blocks['block1_1']
#         self.model2_1 = blocks['block2_1']
#         self.model3_1 = blocks['block3_1']
#         self.model4_1 = blocks['block4_1']
#         self.model5_1 = blocks['block5_1']
#         self.model6_1 = blocks['block6_1']
#
#         self.model1_2 = blocks['block1_2']
#         self.model2_2 = blocks['block2_2']
#         self.model3_2 = blocks['block3_2']
#         self.model4_2 = blocks['block4_2']
#         self.model5_2 = blocks['block5_2']
#         self.model6_2 = blocks['block6_2']
#
#     def make_layers(self, block, no_relu_layers):
#         layers = []
#         for layer_name, v in block.items():
#             if 'pool' in layer_name:
#                 layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
#                                      padding=v[2])
#                 layers.append((layer_name, layer))
#             else:
#                 nn.Conv2d = nn.nn.Conv2d(in_channels=v[0], out_channels=v[1],
#                                    kernel_size=v[2], stride=v[3],
#                                    padding=v[4])
#                 layers.append((layer_name, nn.Conv2d))
#                 if layer_name not in no_relu_layers:
#                     layers.append(('relu_'+layer_name, nn.nn.ReLU(inplace=True)))  # noqa
#
#         return nn.Sequential(OrderedDict(layers))
#
#     def forward(self, x):
#
#         out1 = self.model0(x)
#         out1 = self.model0_0(out1)
#
#         out1_1 = self.model1_1(out1)
#         out1_2 = self.model1_2(out1)
#         out2 = torch.cat([out1_1, out1_2, out1], 1)
#
#         out2_1 = self.model2_1(out2)
#         out2_2 = self.model2_2(out2)
#         out3 = torch.cat([out2_1, out2_2, out1], 1)
#
#         out3_1 = self.model3_1(out3)
#         out3_2 = self.model3_2(out3)
#         out4 = torch.cat([out3_1, out3_2, out1], 1)
#
#         out4_1 = self.model4_1(out4)
#         out4_2 = self.model4_2(out4)
#         out5 = torch.cat([out4_1, out4_2, out1], 1)
#
#         out5_1 = self.model5_1(out5)
#         out5_2 = self.model5_2(out5)
#         out6 = torch.cat([out5_1, out5_2, out1], 1)
#
#         out6_1 = self.model6_1(out6)
#         out6_2 = self.model6_2(out6)
#
#         return out6_1, out6_2


def get_model(config: dict):
    model = CocoPoseNet()
    return model


if __name__ == "__main__":
    with open('../configs/config.yml') as f:
        config = yaml.safe_load(f)

    print(CocoPoseNet())
