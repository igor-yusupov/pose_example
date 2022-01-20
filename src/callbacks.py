import os
from typing import List

from catalyst.core.callback import Callback, CallbackOrder
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .utils import PoseDetector


class MeanLossCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Metric)

    def on_loader_start(self, runner: str = "IRunner"):
        self.losses: List = []

    def on_batch_end(self, runner: str = "IRunner"):
        self.losses.append(runner.batch_metrics['loss'])

    def on_loader_end(self, runner: str = "IRunner"):
        loss = torch.mean(torch.Tensor(self.losses))
        # runner.loader_metrics['loss'] = loss

        if runner.is_valid_loader:
            runner.epoch_metrics["valid_loss"] = loss
        else:
            runner.epoch_metrics["train_loss"] = loss


class CustomTensorBoard(Callback):
    def __init__(self, log_path, inference_img_size=368,
                 heatmap_size=320):
        super().__init__(order=CallbackOrder.external)
        self.log_path = log_path
        self.inference_img_size = inference_img_size
        self.heatmap_size = heatmap_size
        self.pose_detector = PoseDetector()

    def on_stage_start(self, runner: str = "IRunner"):
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.log_path,
                                                               'train'))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.log_path,
                                                             'validation'))

        self.val_ds = runner.loaders['valid'].dataset

    def on_epoch_end(self, runner: str = "IRunner"):
        n_iter = runner.global_epoch_step
        train_loss = runner.epoch_metrics["train_loss"]
        valid_loss = runner.epoch_metrics["valid_loss"]
        self.train_writer.add_scalar('Loss', train_loss, n_iter)
        self.val_writer.add_scalar('Loss', valid_loss, n_iter)

    def on_loader_end(self, runner: str = "IRunner"):
        if not runner.is_train_loader:
            for i in range(20):
                # data = self.val_ds[i]
                img_id = self.val_ds.imgIds[i]
                img_path = os.path.join(
                    'coco2017',
                    'val2017', self.val_ds.coco.loadImgs(
                        [img_id])[0]['file_name'])
                res_img = cv2.imread(img_path)
                # x_data = data['img']
                res_img = self.detect(res_img, runner)
                res_img = torch.tensor(res_img.transpose(2, 0, 1))

                self.val_writer.add_image(
                    f'img_{i}', res_img, global_step=runner.global_epoch_step)

    def on_stage_end(self, runner: str = "IRunner"):
        self.train_writer.close()
        self.val_writer.close()

    def unpreprocess(self, x_data):
        x_data = np.array(x_data)
        x_data = x_data.transpose(1, 2, 0)
        x_data += 0.5
        x_data *= 255

        return x_data

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def normilize(self, inputs, mean, std, div_value):
        inputs = inputs.div(div_value)
        for t, m, s in zip(inputs, mean, std):
            t.sub_(m).div_(s)

        return inputs

    def detect(self, img, runner):
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.unpreprocess(x_data)
        orig_img_h, orig_img_w, _ = img.shape
        input_w, input_h = self.pose_detector.compute_optimal_size(
            img, self.inference_img_size)
        map_w, map_h = self.pose_detector.compute_optimal_size(
            img, self.heatmap_size)

        resized_image = cv2.resize(img, (input_w, input_h))
        x_data = self.preprocess(resized_image)

        x_data = torch.tensor(x_data).to(runner.device)
        x_data.requires_grad = False

        with torch.no_grad():

            h1s, h2s = runner.model(x_data.unsqueeze(0).cuda())
            pafs = F.interpolate(
                h1s[-1], (map_h, map_w), mode='bilinear',
                align_corners=True).cpu().numpy()[0]
            heatmaps = F.interpolate(
                h2s[-1], (map_h, map_w), mode='bilinear',
                align_corners=True).cpu().numpy()[0]

        all_peaks = self.pose_detector.compute_peaks_from_heatmaps(
            heatmaps)

        if len(all_peaks) == 0:
            return img

        all_connections = self.pose_detector.compute_connections(
            pafs, all_peaks, map_w)
        subsets = self.pose_detector.grouping_key_points(
            all_connections, all_peaks)

        all_peaks[:, 1] *= orig_img_w / map_w
        all_peaks[:, 2] *= orig_img_h / map_h
        poses = self.pose_detector.subsets_to_pose_array(
            subsets, all_peaks)

        img = self.pose_detector.draw_person_pose(img, poses)

        return img


def get_callbacks(config: dict):
    required_callbacks = config.get("callbacks", None)
    if required_callbacks is None:
        return []
    callbacks = []
    for callback_conf in required_callbacks:
        name = callback_conf["name"]
        params = callback_conf["params"]
        callback_cls = globals().get(name)

        if callback_cls is not None:
            if params is not None:
                callbacks.append(callback_cls(**params))
            else:
                callbacks.append(callback_cls())

    return callbacks
