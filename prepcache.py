import argparse
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

# from src.utils import enumerateWithEstimate
from src.datasets import CocoDataset


data_dir = 'COCO_KPTS'
dataset_conf = {
    'data_dir': data_dir,
    'input_size': 368,
    'mode': 'train',
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


class COCOPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--batch-size',
            help='Batch size to use for training',
            default=1,
            type=int,
        )
        parser.add_argument(
            '--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        self.prep_dl = DataLoader(
            CocoDataset(dataset_conf),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        for _ in tqdm(self.prep_dl):
            pass


if __name__ == '__main__':
    COCOPrepCacheApp().main()
