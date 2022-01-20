import os
from enum import IntEnum
import random

import argparse
import cv2
import torch
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import yaml


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    return parser


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


class JointType(IntEnum):
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17


class PoseDetector:
    def __init__(self, gaussian_sigma=2.5, heatmap_peak_thresh=0.05):
        self.gaussian_sigma = gaussian_sigma
        self.heatmap_peak_thresh = heatmap_peak_thresh
        self.limbs_point = [
            [JointType.Neck, JointType.RightWaist],
            [JointType.RightWaist, JointType.RightKnee],
            [JointType.RightKnee, JointType.RightFoot],
            [JointType.Neck, JointType.LeftWaist],
            [JointType.LeftWaist, JointType.LeftKnee],
            [JointType.LeftKnee, JointType.LeftFoot],
            [JointType.Neck, JointType.RightShoulder],
            [JointType.RightShoulder, JointType.RightElbow],
            [JointType.RightElbow, JointType.RightHand],
            [JointType.RightShoulder, JointType.RightEar],
            [JointType.Neck, JointType.LeftShoulder],
            [JointType.LeftShoulder, JointType.LeftElbow],
            [JointType.LeftElbow, JointType.LeftHand],
            [JointType.LeftShoulder, JointType.LeftEar],
            [JointType.Neck, JointType.Nose],
            [JointType.Nose, JointType.RightEye],
            [JointType.Nose, JointType.LeftEye],
            [JointType.RightEye, JointType.RightEar],
            [JointType.LeftEye, JointType.LeftEar]
        ]
        self.params = {
            'limbs_point': self.limbs_point,
            'n_integ_points': 10,
            'limb_length_ratio': 1.0,
            'length_penalty_value': 1,
            'n_integ_points_thresh': 8,
            'inner_product_thresh': 0.05,
            'n_subset_limbs_thresh': 3,
            'subset_score_thresh': 0.2,
        }

    def draw_person_pose(self, orig_img, poses):
        if len(poses) == 0:
            return orig_img

        limb_colors = [
            [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0],
            [255, 255, 0.],
            [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.],
            [0, 0, 255],
            [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
        ]

        joint_colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
            [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
            [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        canvas = orig_img.copy()

        # limbs
        for pose in poses.round().astype('i'):
            for i, (limb, color) in enumerate(
                    zip(self.limbs_point, limb_colors)):
                if i != 9 and i != 13:  # don't show ear-shoulder connection
                    limb_ind = np.array(limb)
                    if np.all(pose[limb_ind][:, 2] != 0):
                        joint1, joint2 = pose[limb_ind][:, :2]
                        cv2.line(
                            canvas, tuple(joint1), tuple(joint2), color, 2)

        # joints
        for pose in poses.round().astype('i'):
            for i, ((x, y, v), color) in enumerate(zip(pose, joint_colors)):
                if v != 0:
                    cv2.circle(canvas, (x, y), 3, color, -1)
        return canvas

    def compute_peaks_from_heatmaps(self, heatmaps):
        """
        all_peaks: shape = [N, 5], column = (jointtype, x, y, score, index)
        """

        heatmaps = heatmaps[:-1]

        all_peaks = []
        peak_counter = 0
        for i, heatmap in enumerate(heatmaps):
            heatmap = gaussian_filter(heatmap, sigma=self.gaussian_sigma)
            map_left = np.zeros(heatmap.shape)
            map_right = np.zeros(heatmap.shape)
            map_top = np.zeros(heatmap.shape)
            map_bottom = np.zeros(heatmap.shape)
            map_left[1:, :] = heatmap[:-1, :]
            map_right[:-1, :] = heatmap[1:, :]
            map_top[:, 1:] = heatmap[:, :-1]
            map_bottom[:, :-1] = heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce((
                heatmap > self.heatmap_peak_thresh,
                heatmap > map_left,
                heatmap > map_right,
                heatmap > map_top,
                heatmap > map_bottom,
            ))
            # [(x, y), (x, y)...]のpeak座標配列
            peaks = zip(
                np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])

            peaks_with_score = [
                (i,) + peak_pos + (
                    heatmap[peak_pos[1], peak_pos[0]],) for peak_pos in peaks]
            peaks_id = range(
                peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [
                peaks_with_score[i] + (peaks_id[i], ) for i in range(
                    len(peaks_id))]
            peak_counter += len(peaks_with_score_and_id)
            all_peaks.append(peaks_with_score_and_id)
        all_peaks = np.array(
            [peak for peaks_each_category in all_peaks
             for peak in peaks_each_category])

        return all_peaks

    def compute_connections(self, pafs, all_peaks, img_len):
        all_connections = []
        for i in range(len(self.params['limbs_point'])):
            paf_index = [i*2, i*2 + 1]
            paf = pafs[paf_index]
            limb_point = self.params['limbs_point'][i]
            cand_a = all_peaks[all_peaks[:, 0] == limb_point[0]][:, 1:]
            cand_b = all_peaks[all_peaks[:, 0] == limb_point[1]][:, 1:]

            if len(cand_a) > 0 and len(cand_b) > 0:
                candidate_connections = self.compute_candidate_connections(
                    paf, cand_a, cand_b, img_len)
                connections = np.zeros((0, 3))
                for index_a, index_b, score in candidate_connections:
                    if (
                            index_a not in connections[:, 0] and
                            index_b not in connections[:, 1]):
                        connections = np.vstack(
                            [connections, [index_a, index_b, score]])
                        if len(connections) >= min(len(cand_a), len(cand_b)):
                            break
                all_connections.append(connections)
            else:
                all_connections.append(np.zeros((0, 3)))
        return all_connections

    def compute_candidate_connections(self, paf, cand_a, cand_b, img_len):
        candidate_connections = []
        for joint_a in cand_a:
            for joint_b in cand_b:  # jointは(x, y)座標
                vector = joint_b[:2] - joint_a[:2]
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue

                ys = np.linspace(
                    joint_a[1], joint_b[1], num=self.params['n_integ_points'])
                xs = np.linspace(
                    joint_a[0], joint_b[0], num=self.params['n_integ_points'])
                integ_points = np.stack([ys, xs]).T.round().astype('i')
                paf_in_edge = np.hstack(
                    [paf[0][np.hsplit(integ_points, 2)],
                     paf[1][np.hsplit(integ_points, 2)]])
                unit_vector = vector / norm
                inner_products = np.dot(paf_in_edge, unit_vector)

                integ_value = inner_products.sum() / len(inner_products)
                # vectorの長さが基準値以上の時にペナルティを与える
                integ_value_with_dist_prior = integ_value + min(
                    self.params['limb_length_ratio'] * img_len / norm -
                    self.params['length_penalty_value'], 0)

                n_valid_points = sum(inner_products > self.params[
                    'inner_product_thresh'])
                if (n_valid_points > self.params['n_integ_points_thresh'] and
                        integ_value_with_dist_prior > 0):
                    candidate_connections.append(
                        [int(joint_a[3]), int(joint_b[3]),
                         integ_value_with_dist_prior])
        candidate_connections = sorted(
            candidate_connections, key=lambda x: x[2], reverse=True)
        return candidate_connections

    def subsets_to_pose_array(self, subsets, all_peaks):
        person_pose_array = []
        for subset in subsets:
            joints = []
            for joint_index in subset[:18].astype('i'):
                if joint_index >= 0:
                    joint = all_peaks[joint_index][1:3].tolist()
                    joint.append(2)
                    joints.append(joint)
                else:
                    joints.append([0, 0, 0])
            person_pose_array.append(np.array(joints))
        person_pose_array = np.array(person_pose_array)
        return person_pose_array

    def grouping_key_points(self, all_connections, candidate_peaks):
        subsets = -1 * np.ones((0, 20))

        for i, connections in enumerate(all_connections):
            joint_a, joint_b = self.params['limbs_point'][i]

            for ind_a, ind_b, score in connections[:, :3]:
                ind_a, ind_b = int(ind_a), int(ind_b)

                joint_found_cnt = 0
                joint_found_subset_index = [-1, -1]
                for subset_ind, subset in enumerate(subsets):
                    # そのconnectionのjointをもってるsubsetがいる場合
                    if subset[joint_a] == ind_a or subset[joint_b] == ind_b:
                        joint_found_subset_index[joint_found_cnt] = subset_ind
                        joint_found_cnt += 1

                if joint_found_cnt == 1:  # そのconnectionのどちらかのjointをsubsetが持っている場合  # noqa
                    found_subset = subsets[joint_found_subset_index[0]]
                    # 肩->耳のconnectionの組合せを除いて、始点の一致しか起こり得ない。肩->耳の場合、終点が一致していた場合は、既に顔のbone検出済みなので処理不要。
                    if found_subset[joint_b] != ind_b:
                        found_subset[joint_b] = ind_b
                        found_subset[-1] += 1  # increment joint count
                        found_subset[-2] += candidate_peaks[ind_b, 3] + score   # joint bのscoreとconnectionの積分値を加算  # noqa

                elif joint_found_cnt == 2:  # subset1にjoint1が、subset2にjoint2がある場合(肩->耳のconnectionの組合せした起こり得ない)  # noqa
                    # print('limb {}: 2 subsets have any joint'.format(l))
                    found_subset_1 = subsets[joint_found_subset_index[0]]
                    found_subset_2 = subsets[joint_found_subset_index[1]]

                    membership = (
                        (found_subset_1 >= 0).astype(int) +
                        (found_subset_2 >= 0).astype(int))[:-2]
                    # merge two subsets when no duplication
                    if not np.any(membership == 2):
                        # default is -1
                        found_subset_1[:-2] += found_subset_2[:-2] + 1
                        found_subset_1[-2:] += found_subset_2[-2:]
                        # connectionの積分値のみ加算(jointのscoreはmerge時に全て加算済み)
                        found_subset_1[-2:] += score
                        subsets = np.delete(
                            subsets, joint_found_subset_index[1], axis=0)
                    else:
                        if found_subset_1[joint_a] == -1:
                            found_subset_1[joint_a] = ind_a
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_a, 3] + score  # noqa
                        elif found_subset_1[joint_b] == -1:
                            found_subset_1[joint_b] = ind_b
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_b, 3] + score  # noqa
                        if found_subset_2[joint_a] == -1:
                            found_subset_2[joint_a] = ind_a
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_a, 3] + score  # noqa
                        elif found_subset_2[joint_b] == -1:
                            found_subset_2[joint_b] = ind_b
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_b, 3] + score  # noqa

                # 新規subset作成, 肩耳のconnectionは新規group対象外
                elif joint_found_cnt == 0 and i != 9 and i != 13:
                    row = -1 * np.ones(20)
                    row[joint_a] = ind_a
                    row[joint_b] = ind_b
                    row[-1] = 2
                    row[-2] = sum(candidate_peaks[[ind_a, ind_b], 3]) + score
                    subsets = np.vstack([subsets, row])
                elif joint_found_cnt >= 3:
                    pass

        # delete low score subsets
        keep = np.logical_and(
            subsets[:, -1] >= self.params['n_subset_limbs_thresh'],
            subsets[:, -2]/subsets[:, -1] >= self.params[
                'subset_score_thresh'])
        subsets = subsets[keep]
        return subsets

    def compute_optimal_size(self, orig_img, img_size, stride=8):
        orig_img_h, orig_img_w, _ = orig_img.shape
        aspect = orig_img_h / orig_img_w
        if orig_img_h < orig_img_w:
            img_h = img_size
            img_w = np.round(img_size / aspect).astype(int)
            surplus = img_w % stride
            if surplus != 0:
                img_w += stride - surplus
        else:
            img_w = img_size
            img_h = np.round(img_size * aspect).astype(int)
            surplus = img_h % stride
            if surplus != 0:
                img_h += stride - surplus
        return (img_w, img_h)

    def preprocess(self, img):
        # x_data = img.astype('f')
        x_data = img
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data
