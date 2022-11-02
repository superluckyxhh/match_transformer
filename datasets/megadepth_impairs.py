import torch
import os
import cv2
import numpy as np
from collections import OrderedDict
from pathlib import Path
import deepdish as dd


def array_to_tensor(img_array):
    # img_array = torch.FloatTensor(img_array / 255.0).unsqueeze(0)
    img_array = torch.FloatTensor(img_array / 255.0).permute(2, 0, 1)
    # img_array = torch.FloatTensor(img_array).permute(2, 0, 1)
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # img_array = (img_array - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    return img_array


class BaseMegaDepthPairsDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, scenes_list, max_pairs_per_scene=None, overlap=None):
        self.root_path = Path(root_path)

        pairs_metadata_files = {scene: self.root_path / 'pairs' / scene / 'sparse-txt' / 'pairs.txt' for scene
                                in scenes_list}
        self.image_pairs = OrderedDict()
        for scene, pairs_path in pairs_metadata_files.items():
            try:
                with open(pairs_path) as f:
                    pairs_metadata = f.readlines()
                    pairs_metadata = list(map(lambda x: x.rstrip(), pairs_metadata))
                    if overlap is not None:  # keep pairs with given overlap
                        pairs_metadata = self.filter_pairs_by_overlap(pairs_metadata, overlap)
            except FileNotFoundError:
                pairs_metadata = []
            self.image_pairs[scene] = pairs_metadata
        self.scene_pairs_numbers = OrderedDict([(k, len(v)) for k, v in self.image_pairs.items()])

        if max_pairs_per_scene is not None:  # validation
            self.scene_pairs_numbers = {k: min(v, max_pairs_per_scene) for k, v in self.scene_pairs_numbers.items()}

    def __len__(self):
        return sum(self.scene_pairs_numbers.values())

    def __getitem__(self, idx):
        for s, pairs_num in self.scene_pairs_numbers.items():
            if idx < pairs_num:
                scene, scene_idx = s, idx
                break
            else:
                idx -= pairs_num
        metadata = self.image_pairs[scene][scene_idx]
        return self.parse_pairs_line(metadata), scene, scene_idx

    @staticmethod
    def parse_pairs_line(line):
        img0_name, img1_name, _, _, *camera_params, overlap = line.split(' ')
        camera_params = list(map(lambda x: float(x), camera_params))
        K0, K1, RT = camera_params[:9], camera_params[9:18], camera_params[18:]
        K0 = np.array(K0).astype(np.float32).reshape(3, 3)
        K1 = np.array(K1).astype(np.float32).reshape(3, 3)
        RT = np.array(RT).astype(np.float32).reshape(4, 4)
        R, T = RT[:3, :3], RT[:3, 3]
        return img0_name, img1_name, K0, K1, R, T, float(overlap)

    @staticmethod
    def filter_pairs_by_overlap(pairs_metadata, overlap_range):
        result = []
        min_overlap, max_overlap = overlap_range
        for line in pairs_metadata:
            overlap = float(line.split(' ')[-1])
            if min_overlap <= overlap <= max_overlap:
                result.append(line)
        return result


class MegaDepthPairsDataset(BaseMegaDepthPairsDataset):
    def __init__(self, root_path, scenes_list, target_size=None, random_crop=False, max_pairs_per_scene=None,
                 overlap=None):
        super(MegaDepthPairsDataset, self).__init__(root_path, scenes_list, max_pairs_per_scene, overlap)
        self.target_size = tuple(target_size) if target_size is not None else None
        self.random_crop = random_crop

    def __getitem__(self, idx):
        (img0_name, img1_name, K0, K1, R, T, overlap), \
        scene, scene_idx = super(MegaDepthPairsDataset, self).__getitem__(idx)

        # read and transform images
        images = []
        for img_name, K in ((img0_name, K0), (img1_name, K1)):
            image = cv2.imread(str(self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/imgs' / img_name))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            depth = dd.io.load(str(
                self.root_path / 'phoenix/S6/zl548/MegaDepth_v1' / scene / 'dense0/depths' / (img_name[:-3] + 'h5')))[
                'depth']

            if self.target_size is not None:
                size = image.shape[:2][::-1]
                current_ratio = size[0] / size[1]
                target_ratio = self.target_size[0] / self.target_size[1]

                if current_ratio > target_ratio:
                    resize_height = self.target_size[1]
                    resize_width = int(current_ratio * resize_height)

                    image = cv2.resize(image, (resize_width, resize_height))
                    depth = cv2.resize(depth, (resize_width, resize_height), cv2.INTER_NEAREST)
                    # crop width
                    # max fixes case when resize_width == self.target_size[0]
                    if self.random_crop:
                        start_width = np.random.randint(0, max(resize_width - self.target_size[0], 1))
                    else:
                        start_width = (resize_width - self.target_size[0]) // 2
                    end_width = start_width + self.target_size[0]

                    image = image[:, start_width:end_width]
                    depth = depth[:, start_width:end_width]
                    # update K
                    scales = np.diag([resize_width / size[0], resize_height / size[1], 1.0]).astype(np.float32)
                    K = np.dot(scales, K)
                    K[0, 2] -= start_width
                else:
                    resize_width = self.target_size[0]
                    resize_height = int(resize_width / current_ratio)

                    image = cv2.resize(image, (resize_width, resize_height))
                    depth = cv2.resize(depth, (resize_width, resize_height), cv2.INTER_NEAREST)
                    # crop height
                    if self.random_crop:
                        start_height = np.random.randint(0, max(resize_height - self.target_size[1], 1))
                    else:
                        start_height = (resize_height - self.target_size[1]) // 2
                    end_height = start_height + self.target_size[1]

                    image = image[start_height:end_height, :]
                    depth = depth[start_height:end_height, :]
                    # update K
                    scales = np.diag([resize_width / size[0], resize_height / size[1], 1.0]).astype(np.float32)
                    K = np.dot(scales, K)
                    K[1, 2] -= start_height

            images.append((image, depth, K))

        (image0, depth0, K0), (image1, depth1, K1) = images

        transformation = {
            'type': '3d_reprojection',
            'K0': torch.from_numpy(K0),
            'K1': torch.from_numpy(K1),
            'R': torch.from_numpy(R),
            'T': torch.from_numpy(T),
            'depth0': torch.from_numpy(depth0),
            'depth1': torch.from_numpy(depth1),
        }

        return {'image0': array_to_tensor(image0), 'image1': array_to_tensor(image1), 'transformation': transformation}