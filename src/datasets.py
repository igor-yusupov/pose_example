import os

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from tools.json_helper import JsonHelper
from tools.image_helper import ImageHelper
from tools.logger import Logger as Log

from src.augmentations import Augmentations
from src.utils import JointType


class CocoDataset2(Dataset):
    def __init__(self, conf):
        self.conf = conf
        if self.conf['mode'] == 'train':
            self.coco = COCO(os.path.join(
                self.conf['data_dir'],
                'annotations/person_keypoints_train2017.json'))
        else:
            self.coco = COCO(os.path.join(
                self.conf['data_dir'],
                'annotations/person_keypoints_val2017.json'))
        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.imgIds = sorted(self.coco.getImgIds(catIds=self.catIds))
        print('{} images: {}'.format(self.conf['mode'], len(self)))
        self.insize = self.conf['input_size']
        self.augment = Augmentations(conf)
        self.coco_joint_indices = [
            JointType.Nose,
            JointType.LeftEye,
            JointType.RightEye,
            JointType.LeftEar,
            JointType.RightEar,
            JointType.LeftShoulder,
            JointType.RightShoulder,
            JointType.LeftElbow,
            JointType.RightElbow,
            JointType.LeftHand,
            JointType.RightHand,
            JointType.LeftWaist,
            JointType.RightWaist,
            JointType.LeftKnee,
            JointType.RightKnee,
            JointType.LeftFoot,
            JointType.RightFoot
        ]
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

    def __len__(self):
        return len(self.imgIds)

    def generate_gaussian_heatmap(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_heatmaps(self, img, poses, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for pose in poses:
                if pose[joint_index, 2] > 0:
                    jointmap = self.generate_gaussian_heatmap(
                        img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[
                        jointmap > sum_heatmap]
            heatmaps = np.vstack(
                (heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))

        return heatmaps.astype('f')

    # return shape: (2, height, width)
    def generate_constant_paf(self, shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to):
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array(
            [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        horizontal_inner_product = (
            unit_vector[0] * (grid_x - joint_from[0]) +
            unit_vector[1] * (grid_y - joint_from[1]))
        horizontal_paf_flag = (
            (0 <= horizontal_inner_product) &
            (horizontal_inner_product <= joint_distance))

        vertical_inner_product = (
            vertical_unit_vector[0] * (grid_x - joint_from[0]) +
            vertical_unit_vector[1] * (grid_y - joint_from[1]))
        vertical_paf_flag = np.abs(
            vertical_inner_product) <= paf_width

        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack(
            (paf_flag, paf_flag)) * np.broadcast_to(
                unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)

        return constant_paf

    def generate_pafs(self, img, poses, paf_sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in self.limbs_point:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape)

            for pose in poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0:
                    limb_paf = self.generate_constant_paf(
                        img.shape, joint_from[:2], joint_to[:2],
                        paf_sigma)
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(
                        limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)

                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def generate_labels(self, img, poses, ignore_mask):
        img, ignore_mask, poses = self.augment.augment_data(
            img, ignore_mask, poses)
        resized_img, ignore_mask, resized_poses = self.augment.resize_data(
            img, ignore_mask, poses, shape=(self.conf['input_size'],
                                            self.conf['input_size']))
        heatmaps = self.generate_heatmaps(
            resized_img, resized_poses, self.conf['heatmap_sigma'])
        pafs = self.generate_pafs(
            resized_img, resized_poses, self.conf['paf_sigma'])
        ignore_mask = cv2.morphologyEx(
            ignore_mask.astype('uint8'), cv2.MORPH_DILATE,
            np.ones((16, 16))).astype('bool')
        return resized_img, pafs, heatmaps, ignore_mask

    def get_img_annotation(self, ind=None, img_id=None):
        annotations = None

        if ind is not None:
            img_id = self.imgIds[ind]
        anno_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)

        # annotation for that image
        if len(anno_ids) > 0:
            annotations_for_img = self.coco.loadAnns(anno_ids)

            person_cnt = 0
            valid_annotations_for_img = []
            for annotation in annotations_for_img:
                # if too few keypoints or too small
                if (annotation['num_keypoints'] >= self.conf['min_keypoints']
                        and annotation['area'] > self.conf['min_area']):
                    person_cnt += 1
                    valid_annotations_for_img.append(annotation)

            # if person annotation
            if person_cnt > 0:
                annotations = valid_annotations_for_img

        if self.conf['mode'] == 'train':
            img_path = os.path.join(
                self.conf['data_dir'], 'train2017',
                self.coco.loadImgs([img_id])[0]['file_name'])
            mask_path = os.path.join(
                self.conf['data_dir'], 'ignore_mask_train2017',
                '{:012d}.png'.format(img_id))
        else:
            img_path = os.path.join(
                self.conf['data_dir'], 'val2017',
                self.coco.loadImgs([img_id])[0]['file_name'])
            mask_path = os.path.join(
                self.conf['data_dir'], 'ignore_mask_val2017',
                '{:012d}.png'.format(img_id))
        img = cv2.imread(img_path)
        ignore_mask = cv2.imread(mask_path, 0)
        if ignore_mask is None:
            ignore_mask = np.zeros(img.shape[:2], 'bool')
        else:
            ignore_mask = ignore_mask == 255

        if self.conf['mode'] == 'eval':
            return img, img_id, annotations_for_img, ignore_mask
        return img, img_id, annotations, ignore_mask

    def parse_coco_annotation(self, annotations):
        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)

        for ann in annotations:
            ann_pose = np.array(ann['keypoints']).reshape(-1, 3)
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert poses position
            for i, joint_index in enumerate(self.coco_joint_indices):
                pose[0][joint_index] = ann_pose[i]

            # compute neck position
            if pose[0][JointType.LeftShoulder][2] > 0 and pose[0][
                    JointType.RightShoulder][2] > 0:
                pose[0][JointType.Neck][0] = int(
                    (pose[0][JointType.LeftShoulder][0] + pose[0][
                        JointType.RightShoulder][0]) / 2)
                pose[0][JointType.Neck][1] = int(
                    (pose[0][JointType.LeftShoulder][1] + pose[0][
                        JointType.RightShoulder][1]) / 2)
                pose[0][JointType.Neck][2] = 2

            poses = np.vstack((poses, pose))

#         gt_pose = np.array(ann['keypoints']).reshape(-1, 3)
        return poses

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def __getitem__(self, idx):
        img, img_id, annotations, ignore_mask = self.get_img_annotation(
            ind=idx)

        if self.conf['mode'] == 'eval':
            # don't need to make heatmaps/pafs
            return img, annotations, img_id

        # if no annotations are available
        while annotations is None:
            img_id = self.imgIds[np.random.randint(len(self))]
            img, img_id, annotations, ignore_mask = self.get_img_annotation(
                img_id=img_id)

        poses = self.parse_coco_annotation(annotations)
        resized_img, pafs, heatmaps, ignore_mask = self.generate_labels(
            img, poses, ignore_mask)
        resized_img = self.preprocess(resized_img)
        resized_img = torch.tensor(resized_img)
        pafs = torch.tensor(pafs)
        heatmaps = torch.tensor(heatmaps)
        ignore_mask = torch.tensor(ignore_mask.astype('f'))
        return dict(
            img=resized_img,
            pafs=pafs,
            heatmaps=heatmaps,
            ignore_mask=ignore_mask)


class CocoDataset(Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.img_list, self.json_list, self.mask_list = self.__list_dirs(
            self.conf['data_dir'], self.conf['mode'])

        self.augment = Augmentations(conf)

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

    def __list_dirs(self, data_dir, mode):
        img_list = list()
        json_list = list()
        mask_list = list()
        image_dir = os.path.join(data_dir, mode, 'image')
        json_dir = os.path.join(data_dir, mode, 'json')
        mask_dir = os.path.join(data_dir, mode, 'mask')

        for file_name in os.listdir(json_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            mask_path = os.path.join(mask_dir, '{}.png'.format(image_name))
            img_path = ImageHelper.imgpath(image_dir, image_name)
            json_path = os.path.join(json_dir, file_name)
            if not os.path.exists(json_path) or img_path is None:
                Log.warn('Json Path: {} not exists.'.format(json_path))
                continue

            json_list.append(json_path)
            mask_list.append(mask_path)
            img_list.append(img_path)

        if mode == 'train' and self.conf['include_val']:
            image_dir = os.path.join(data_dir, 'val/image')
            json_dir = os.path.join(data_dir, 'val/json')
            mask_dir = os.path.join(data_dir, 'val/mask')
            for file_name in os.listdir(json_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                mask_path = os.path.join(mask_dir, '{}.png'.format(image_name))
                img_path = ImageHelper.imgpath(image_dir, image_name)
                json_path = os.path.join(json_dir, file_name)
                if not os.path.exists(json_path) or img_path is None:
                    Log.warn('Json Path: {} not exists.'.format(json_path))
                    continue

                json_list.append(json_path)
                mask_list.append(mask_path)
                img_list.append(img_path)

        return img_list, json_list, mask_list

    def __read_json_file(self, json_file):
        json_dict = JsonHelper.load_file(json_file)

        kpts = list()
        bboxes = list()

        for object in json_dict['objects']:
            kpts.append(object['kpts'])
            if 'bbox' in object:
                bboxes.append(object['bbox'])

        return np.array(kpts).astype(
            np.float32), np.array(bboxes).astype(np.float32)

    def generate_labels(self, img, poses, ignore_mask):
        # img, ignore_mask, poses = self.augment.augment_data(
        #     img, ignore_mask, poses)
        resized_img, ignore_mask, resized_poses = self.augment.resize_data(
            img, ignore_mask, poses, shape=(self.conf['input_size'],
                                            self.conf['input_size']))
        heatmaps = self.generate_heatmaps(
            resized_img, resized_poses, self.conf['heatmap_sigma'])
        pafs = self.generate_pafs(
            resized_img, resized_poses, self.conf['paf_sigma'])
        ignore_mask = cv2.morphologyEx(
            ignore_mask.astype('uint8'),
            cv2.MORPH_DILATE, np.ones((16, 16))).astype('bool')

        return resized_img, pafs, heatmaps, ignore_mask

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def resize_label(self, label):
        label_shape = (label.shape[1] // self.conf['stride'],
                       label.shape[2] // self.conf['stride'])

        label = label.unsqueeze(0)
        label = F.interpolate(
            label, label_shape, mode='bilinear', align_corners=True)
        label = label.squeeze(0)

        return label

    def generate_heatmaps(self, img, poses, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for pose in poses:
                if pose[joint_index, 2] > -1:
                    jointmap = self.generate_gaussian_heatmap(
                        img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[
                        jointmap > sum_heatmap]
            heatmaps = np.vstack(
                (heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        # heatmaps = self.resize_label(torch.tensor(heatmaps.astype('f')))
        #
        # return heatmaps
        return heatmaps.astype('f')

    # return shape: (height, width)
    def generate_gaussian_heatmap(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_pafs(self, img, poses, paf_sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in self.limbs_point:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape)  # for constant paf

            for pose in poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > -1 and joint_to[2] > -1:
                    limb_paf = self.generate_constant_paf(
                        img.shape, joint_from[:2], joint_to[:2], paf_sigma)
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(
                        limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        # pafs = self.resize_label(torch.tensor(pafs.astype('f')))
        # return pafs
        return pafs.astype('f')

    # return shape: (2, height, width)
    def generate_constant_paf(self, shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to):  # same joint
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array(
            [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose()
        horizontal_inner_product = (
            unit_vector[0] * (
                grid_x - joint_from[0]) + unit_vector[1] * (
                    grid_y - joint_from[1]))
        horizontal_paf_flag = (
            0 <= horizontal_inner_product) & (
                horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (
            grid_x - joint_from[0]) + vertical_unit_vector[1] * (
                grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width
        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(
            unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)
        return constant_paf

    def normilize(self, inputs, mean, std, div_value):
        inputs = inputs.div(div_value)
        for t, m, s in zip(inputs, mean, std):
            t.sub_(m).div_(s)

        return inputs

    def __len__(self):
        return 20
        # return len(self.img_list)

    def __getitem__(self, idx):
        img = ImageHelper.read_image(
            self.img_list[idx],
            tool=self.conf['image_tool'],
            mode=self.conf['input_mode'])

        if os.path.exists(self.mask_list[idx]):
            # maskmap = ImageHelper.read_image(
            #     self.mask_list[idx],
            #     tool=self.conf['image_tool'],
            #     mode='P')
            ignore_mask = cv2.imread(self.mask_list[idx], 0)
        else:
            # maskmap = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
            # if self.conf['image_tool'] == 'pil':
            #     maskmap = ImageHelper.to_img(maskmap)
            ignore_mask = np.zeros(img.shape[:2], 'bool')

        kpts, bboxes = self.__read_json_file(self.json_list[idx])

        resized_img, pafs, heatmaps, ignore_mask = self.generate_labels(
            img, kpts, ignore_mask)

        img_tensor = self.preprocess(resized_img)
        img_tensor = torch.tensor(img_tensor)
        pafs = torch.tensor(pafs)
        heatmaps = torch.tensor(heatmaps)
        ignore_mask = torch.tensor(ignore_mask.astype('f'))
        # img_tensor = self.normilize(
        #     torch.tensor(resized_img.transpose(2, 0, 1)), div_value=255,
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return dict(
            img=img_tensor,
            pafs=pafs,
            heatmaps=heatmaps,
            ignore_mask=ignore_mask)


if __name__ == '__main__':
    mode = 'train'
    data_dir = 'coco2017'
    dataset_conf = {
        'data_dir': data_dir,
        'input_size': 368,
        'mode': mode,
        'min_keypoints': 5,
        'min_area': 32 * 32,
        'paf_sigma': 8,
        'heatmap_sigma': 7,
        'min_box_size': 64,
        'max_box_size': 512,
        'min_scale': 0.5,
        'max_scale': 2.0,
        'max_rotate_degree': 40,
        'center_perterb_max': 40,
        'stride': 8,
        'image_tool': 'cv2',
        'input_mode': 'RGB',
        'include_val': False
    }
    ds = CocoDataset2(dataset_conf)

    for d in tqdm(ds):
        print(d['heatmaps'].shape)
