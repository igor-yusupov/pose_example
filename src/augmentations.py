import math

import cv2
import numpy as np
import random

from .utils import JointType


class Augmentations:
    def __init__(self, conf):
        self.conf = conf

    def get_pose_bboxes(self, poses):
        pose_bboxes = []
        for pose in poses:
            x1 = pose[pose[:, 2] > 0][:, 0].min()
            y1 = pose[pose[:, 2] > 0][:, 1].min()
            x2 = pose[pose[:, 2] > 0][:, 0].max()
            y2 = pose[pose[:, 2] > 0][:, 1].max()
            pose_bboxes.append([x1, y1, x2, y2])
        pose_bboxes = np.array(pose_bboxes)
        return pose_bboxes

    def random_crop_img(self, img, ignore_mask, poses):
        h, w, _ = img.shape
        insize = self.conf['input_size']
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox = random.choice(joint_bboxes)  # select a bbox randomly
        bbox_center = bbox[:2] + (bbox[2:] - bbox[:2])/2

        r_xy = np.random.rand(2)
        perturb = ((r_xy - 0.5) * 2 * self.conf['center_perterb_max'])
        center = (bbox_center + perturb + 0.5).astype('i')

        crop_img = np.zeros((insize, insize, 3), 'uint8') + 127.5
        crop_mask = np.zeros((insize, insize), 'bool')

        offset = (center - (insize-1)/2 + 0.5).astype('i')
        offset_ = (center + (insize-1)/2 - (w-1, h-1) + 0.5).astype('i')

        x1, y1 = (center - (insize-1)/2 + 0.5).astype('i')
        x2, y2 = (center + (insize-1)/2 + 0.5).astype('i')

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)

        x_from = -offset[0] if offset[0] < 0 else 0
        y_from = -offset[1] if offset[1] < 0 else 0
        x_to = insize - offset_[0] - 1 if offset_[0] >= 0 else insize - 1
        y_to = insize - offset_[1] - 1 if offset_[1] >= 0 else insize - 1

        crop_img[y_from:y_to+1, x_from:x_to+1] = img[y1:y2+1, x1:x2+1].copy()
        crop_mask[y_from:y_to+1, x_from:x_to+1] = ignore_mask[
            y1:y2+1, x1:x2+1].copy()

        poses[:, :, :2] -= offset
        return crop_img.astype('uint8'), crop_mask, poses

    def random_rotate_img(self, img, mask, poses):
        h, w, _ = img.shape
        # degree = (random.random() - 0.5) * 2 * params['max_rotate_degree']
        degree = np.random.randn() / 3 * self.conf['max_rotate_degree']
        rad = degree * math.pi / 180
        center = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(center, degree, 1)
        bbox = (w*abs(math.cos(rad)) + h*abs(math.sin(rad)),
                w*abs(math.sin(rad)) + h*abs(math.cos(rad)))
        R[0, 2] += bbox[0] / 2 - center[0]
        R[1, 2] += bbox[1] / 2 - center[1]
        rotate_img = cv2.warpAffine(
            img, R, (int(bbox[0]+0.5), int(bbox[1]+0.5)),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
            borderValue=[127.5, 127.5, 127.5])
        rotate_mask = cv2.warpAffine(
            mask.astype('uint8')*255, R,
            (int(bbox[0]+0.5), int(bbox[1]+0.5))) > 0

        tmp_poses = np.ones_like(poses)
        tmp_poses[:, :, :2] = poses[:, :, :2].copy()
        # apply rotation matrix to the poses
        tmp_rotate_poses = np.dot(tmp_poses, R.T)
        rotate_poses = poses.copy()  # to keep visibility flag
        rotate_poses[:, :, :2] = tmp_rotate_poses
        return rotate_img, rotate_mask, rotate_poses

    def random_resize_img(self, img, ignore_mask, poses):
        h, w, _ = img.shape
        joint_bboxes = self.get_pose_bboxes(poses)
        bbox_sizes = ((joint_bboxes[:, 2:] - joint_bboxes[:, :2] + 1)**2).sum(
            axis=1)**0.5

        min_scale = self.conf['min_box_size']/bbox_sizes.min()
        max_scale = self.conf['max_box_size']/bbox_sizes.max()

        # print(len(bbox_sizes))
        # print('min: {}, max: {}'.format(min_scale, max_scale))

        min_scale = min(max(min_scale, self.conf['min_scale']), 1)
        max_scale = min(max(max_scale, 1), self.conf['max_scale'])

        # print('min: {}, max: {}'.format(min_scale, max_scale))

        scale = float((max_scale - min_scale) * random.random() + min_scale)
        shape = (round(w * scale), round(h * scale))

        # print(scale)

        resized_img, resized_mask, resized_poses = self.resize_data(
            img, ignore_mask, poses, shape)
        return resized_img, resized_mask, poses

    def resize_data(self, img, ignore_mask, poses, shape):
        """resize img, mask and annotations"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, shape)
        # ignore_mask = cv2.resize(
        #     ignore_mask, shape)
        ignore_mask = cv2.resize(
            ignore_mask.astype(np.uint8), shape).astype('bool')
        poses[:, :, :2] = (poses[:, :, :2] * np.array(shape) / np.array(
            (img_w, img_h)))
        return resized_img, ignore_mask, poses

    def distort_color(self, img):
        img_max = np.broadcast_to(
            np.array(255, dtype=np.uint8), img.shape[:-1])
        img_min = np.zeros(img.shape[:-1], dtype=np.uint8)

        hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv_img[:, :, 0] = np.maximum(
            np.minimum(
                hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), img_max),
            img_min)  # hue
        hsv_img[:, :, 1] = np.maximum(
            np.minimum(
                hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), img_max),
            img_min)  # saturation
        hsv_img[:, :, 2] = np.maximum(
            np.minimum(
                hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), img_max),
            img_min)  # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return distorted_img

    def flip_img(self, img, mask, poses):
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
        poses[:, :, 0] = img.shape[1] - 1 - poses[:, :, 0]

        def swap_joints(poses, joint_type_1, joint_type_2):
            tmp = poses[:, joint_type_1].copy()
            poses[:, joint_type_1] = poses[:, joint_type_2]
            poses[:, joint_type_2] = tmp

        swap_joints(poses, JointType.LeftEye, JointType.RightEye)
        swap_joints(poses, JointType.LeftEar, JointType.RightEar)
        swap_joints(poses, JointType.LeftShoulder, JointType.RightShoulder)
        swap_joints(poses, JointType.LeftElbow, JointType.RightElbow)
        swap_joints(poses, JointType.LeftHand, JointType.RightHand)
        swap_joints(poses, JointType.LeftWaist, JointType.RightWaist)
        swap_joints(poses, JointType.LeftKnee, JointType.RightKnee)
        swap_joints(poses, JointType.LeftFoot, JointType.RightFoot)
        return flipped_img, flipped_mask, poses

    def augment_data(self, img, ignore_mask, poses):
        aug_img = img.copy()
        aug_img, ignore_mask, poses = self.random_resize_img(
            aug_img, ignore_mask, poses)
        aug_img, ignore_mask, poses = self.random_rotate_img(
            aug_img, ignore_mask, poses)
        aug_img, ignore_mask, poses = self.random_crop_img(
            aug_img, ignore_mask, poses)

        if np.random.randint(2):
            aug_img = self.distort_color(aug_img)
        if np.random.randint(2):
            aug_img, ignore_mask, poses = self.flip_img(
                aug_img, ignore_mask, poses)

        return aug_img, ignore_mask, poses
