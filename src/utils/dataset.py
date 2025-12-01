import csv
import glob
import os
import cv2
import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict
import torch
import time
from scipy.spatial.transform import Rotation
from src.utils.camera_utils import fov2focal, focal2fov, getProjectionMatrix, getProjectionMatrix_v2
import torch.multiprocessing as mp
from quick_queue import QQueue
import json
from natsort import natsorted

import warnings
warnings.filterwarnings("ignore", message="Input line.*contained no data.*")


try:
    import pyk4a
    from pyk4a import Config, PyK4A
except Exception:
    pass


class RGBDDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        calib = config.Dataset.Calibration
        
        fovx = focal2fov(calib.fx, calib.width)
        fovy = focal2fov(calib.fy, calib.height)
        projmat = getProjectionMatrix_v2(
            znear   =0.01,
            zfar    =100.0,
            fovX    =fovx,
            fovY    =fovy
        ).transpose(0, 1)
        
        self.params = edict({
            "fx"                : calib.fx,
            "fy"                : calib.fy,
            "cx"                : calib.cx,
            "cy"                : calib.cy,
            "width"             : calib.width,
            "height"            : calib.height,
            "fovx"              : fovx,
            "fovy"              : fovy,
            "projection_matrix" : projmat,
            "depth_scale"       : calib.depth_scale
        })
        
        self.intr = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]])
        self.distCoeff = np.array([calib.k1, calib.k2, calib.p1, calib.p2, calib.k3])
        self.xymap = cv2.initUndistortRectifyMap(
            self.intr,
            self.distCoeff,
            np.eye(3),
            self.intr,
            (calib.width, calib.height),
            cv2.CV_32FC1)

        self.mask = (self.xymap[0] > 0) & (self.xymap[1] > 0) & (self.xymap[0] < calib.width) & (self.xymap[1] < calib.height)
        # self.depth_scale = calib.depth_scale
        
        self.pivot = np.eye(4)
        
        # preload config
        self.buffer_size = 8
        # self.buffer = mp.Manager().Queue(maxsize=self.buffer_size) # slow     
        self.buffer = QQueue(maxsize=self.buffer_size, size_bucket_list=1)

    def preload(self):
        frame_id = 0
        while True:
            if not self.buffer.full() and frame_id < self.n_imgs:
                self.buffer.put_bucket(self.__getitem__(frame_id))
                frame_id += 1
            else:
                time.sleep(0.002)

            if frame_id >= self.n_imgs:
                break
            
    
    def get_buffer_frame(self):
        return self.buffer.get_bucket()
    
    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color = np.array(Image.open(color_path))
        
        if self.config.Dataset.type == "scannet":
            color = cv2.resize(color, (self.params.width, self.params.height), interpolation=cv2.INTER_LINEAR)
        
        color = cv2.remap(color, self.xymap[0], self.xymap[1], cv2.INTER_LINEAR)
        
        depth_path = self.depth_paths[idx]
        depth = np.array(Image.open(depth_path))
        # depth = np.array(Image.open(depth_path))[..., None].astype(np.float32) / self.depth_scale
        
        pose = self.poses[idx]
        
        timestamp = self.ts[idx]
        
        return timestamp, color, depth, self.mask[..., None], pose
    
    def __len__(self):
        return self.n_imgs

class TUMParser:
    def __init__(self, root):
        
        image_list = os.path.join(root, "rgb.txt")
        image_data = np.loadtxt(image_list, delimiter=" ", dtype=np.str_, skiprows=0)
        
        depth_list = os.path.join(root, "depth.txt")
        depth_data = np.loadtxt(depth_list, delimiter=" ", dtype=np.str_, skiprows=0)

        pose_list = os.path.join(root, "groundtruth.txt")
        pose_data = np.loadtxt(pose_list, delimiter=" ", dtype=np.str_, skiprows=1).astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose  = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
        
        indicies = [0]
        frame_rate = 32
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.ts, self.color_paths, self.poses, self.depth_paths = [], [], [], []
        
        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths.append(os.path.join(root, image_data[i, 1]))
            self.depth_paths.append(os.path.join(root, depth_data[j, 1]))
            
            trans, quat = pose_data[k, 1:4], pose_data[k, 4:]
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat(quat).as_matrix()
            c2w[:3,  3] = trans
            self.poses.append(np.linalg.inv(c2w))
            
            self.ts.append(tstamp_image[i])

        init_w2c = self.poses[0]
        for i in range(len(self.poses)):
            self.poses[i] = self.poses[i] @ np.linalg.inv(init_w2c)

        self.pivot = init_w2c
        self.n_imgs = len(self.color_paths)        

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

class TUMDataset(RGBDDataset):
    def __init__(self, config):
        super().__init__(config)
        parser = TUMParser(config.Dataset.dataset_path)
        self.color_paths    = parser.color_paths
        self.depth_paths    = parser.depth_paths
        self.poses          = parser.poses
        self.ts             = parser.ts
        self.n_imgs         = len(self.color_paths)
        self.pivot          = parser.pivot
        
        self.preload_process = mp.Process(target=self.preload)
        self.preload_process.start()
        
class ReplicaParser:
    def __init__(self, root):

        self.color_paths = sorted(glob.glob(f"{root}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{root}/results/depth*.png"))
        
        self.poses = []
        with open(os.path.join(root, 'traj.txt'), "r") as f:
            lines = f.readlines()
        for line in lines:
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
        
        init_w2c = self.poses[0]
        for i in range(len(self.poses)):
            self.poses[i] = self.poses[i] @ np.linalg.inv(init_w2c)

        self.pivot = init_w2c
        self.n_imgs = len(self.color_paths)
        self.ts = np.arange(len(self.color_paths)) * 0.050

class ReplicaDataset(RGBDDataset):
    def __init__(self, config):
        super().__init__(config)
        parser = ReplicaParser(config.Dataset.dataset_path)
        self.color_paths    = parser.color_paths
        self.depth_paths    = parser.depth_paths
        self.poses          = parser.poses
        self.ts             = parser.ts
        self.n_imgs         = len(self.color_paths)
        self.pivot          = parser.pivot
        self.preload_process = mp.Process(target=self.preload)
        self.preload_process.start()

class ScanNetPPParser:
    def __init__(self, root, test):
        all_color_paths = sorted(glob.glob(f"{root}/dslr/undistorted_images/*.JPG"))
        all_depth_paths = sorted(glob.glob(f"{root}/dslr/undistorted_depths/*.png"))
                
        all_frame_list_file = os.path.join(root, "dslr/train_test_lists.json")
            
        all_poses_dict = {}
        is_ok_dict = {}
        json_file = os.path.join(root, "dslr/nerfstudio", "transforms_undistorted.json")
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            for item in data["frames"]:
                all_poses_dict[item["file_path"].split('/')[-1].split('.')[0]] = np.array(item["transform_matrix"]).reshape(4, 4)
                is_ok_dict[item["file_path"].split('/')[-1].split('.')[0]] = not item["is_bad"]
            for item in data["test_frames"]:
                all_poses_dict[item["file_path"].split('/')[-1].split('.')[0]] = np.array(item["transform_matrix"]).reshape(4, 4)
                is_ok_dict[item["file_path"].split('/')[-1].split('.')[0]] = not item["is_bad"]

        with open(all_frame_list_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            _frame_names = sorted([path.split('/')[-1].split('.')[0] for path in data["train"]])
            _test_frame_names = sorted([path.split('/')[-1].split('.')[0] for path in data["test"]])        
        
        frame_names = [f for f in _frame_names if f in is_ok_dict and is_ok_dict[f]]
        test_frame_names = [f for f in _test_frame_names if f in is_ok_dict and is_ok_dict[f]]
        
        for k, v in all_poses_dict.items():
            v[:, 1:3] *= -1
            v = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]) @ v
            all_poses_dict[k] = v

        if not test:
            self.color_paths = [path for path in all_color_paths if path.split('/')[-1].split('.')[0] in frame_names]
            self.depth_paths = [path for path in all_depth_paths if path.split('/')[-1].split('.')[0] in frame_names]
            self.poses = [all_poses_dict[frame] for frame in frame_names]
        else:
            self.color_paths = [path for path in all_color_paths if path.split('/')[-1].split('.')[0] in test_frame_names]
            self.depth_paths = [path for path in all_depth_paths if path.split('/')[-1].split('.')[0] in test_frame_names]
            self.poses = [all_poses_dict[frame] for frame in test_frame_names]

        init_c2w = all_poses_dict[frame_names[0]]
        for i in range(len(self.poses)):
            c2w = self.poses[i]
            self.poses[i] = np.linalg.inv(c2w) @ init_c2w
            
        self.pivot = np.linalg.inv(init_c2w)
        self.n_imgs = len(self.color_paths)
        self.ts = np.arange(len(self.color_paths)) * 0.050
        
class ScanNetPP(RGBDDataset):
    def __init__(self, config, test=False):
        super().__init__(config)
        parser = ScanNetPPParser(config.Dataset.dataset_path, test)
        self.color_paths    = parser.color_paths    
        self.depth_paths    = parser.depth_paths
        self.poses          = parser.poses
        self.ts             = parser.ts
        self.n_imgs         = len(self.color_paths)
        self.pivot          = parser.pivot
        self.preload_process = mp.Process(target=self.preload)
        self.preload_process.start()


class AzureKinectParser:
    def __init__(self, root):

        self.color_paths = sorted(glob.glob(f"{root}/color/*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{root}/depth/*.png"))
        
        assert len(self.color_paths) == len(self.depth_paths)
        
        self.n_imgs = len(self.color_paths)
        
        self.poses = []
        for i in range(self.n_imgs):
            self.poses.append(np.eye(4))

        self.pivot = np.eye(4)
        self.ts = np.arange(len(self.color_paths)) * 0.050


class AzureKinectDataset(RGBDDataset):
    def __init__(self, config):
        super().__init__(config)
        parser = AzureKinectParser(config.Dataset.dataset_path)
        self.color_paths    = parser.color_paths
        self.depth_paths    = parser.depth_paths
        self.poses          = parser.poses
        self.ts             = parser.ts
        self.n_imgs         = len(self.color_paths)
        self.pivot          = parser.pivot
        self.preload_process = mp.Process(target=self.preload)
        self.preload_process.start()

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color = np.array(Image.open(color_path))
        
        depth_path = self.depth_paths[idx]
        depth = np.array(Image.open(depth_path))
        # depth = np.array(Image.open(depth_path))[..., None].astype(np.float32) / self.depth_scale
        
        pose = self.poses[idx]
        
        timestamp = self.ts[idx]
        
        color = cv2.resize(color, (self.params.width, self.params.height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.params.width, self.params.height), interpolation=cv2.INTER_NEAREST)
                
        self.mask = np.ones((self.params.height, self.params.width)).astype(bool)

        return timestamp, color, depth, self.mask[..., None], pose
    
class AzureKinectLive(RGBDDataset):
    def __init__(self, config):
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            )
        )
        self.k4a.start()
        # self.depth_scale = 1000

        sensor_wd, sensor_ht = 1280, 720
        sensor_calib = self.k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
        sensor_fx = sensor_calib[0][0]
        sensor_fy = sensor_calib[1][1]
        sensor_cx = sensor_calib[0][2]
        sensor_cy = sensor_calib[1][2]

        calib = config.Dataset.Calibration
        ratio = calib.width * 1.0 / sensor_wd

        wd = int(sensor_wd * ratio)
        ht = int(sensor_ht * ratio)
        fx = ratio * sensor_fx
        fy = ratio * sensor_fy
        cx = ratio * sensor_cx
        cy = ratio * sensor_cy
        
        fovx = focal2fov(fx, wd)
        fovy = focal2fov(fy, ht)
        projection_matrix = getProjectionMatrix_v2(
            znear   =0.01,
            zfar    =100.0,
            fovX    =fovx,
            fovY    =fovy
        ).transpose(0, 1)

        self.params = edict({
            "fx"                : fx,
            "fy"                : fy,
            "cx"                : cx,
            "cy"                : cy,
            "width"             : wd,
            "height"            : ht,
            "fovx"              : fovx,
            "fovy"              : fovy,
            "projection_matrix" : projection_matrix,
            "depth_scale"       : 1000
        })

        self.mask = np.ones((ht, wd)).astype(bool)

    def __getitem__(self, idx):
        capture = self.k4a.get_capture()
        image = capture.color   # type=uint8
        image = image[:, :, 2::-1].copy() # BGRA to RGB
        depth = capture.transformed_depth
        depth = depth.astype(np.int32) / self.depth_scale
        pose = np.eye(4)
        timestamp = capture.color_timestamp_usec / 1.0e6

        image = cv2.resize(image, (self.params.width, self.params.height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.params.width, self.params.height), interpolation=cv2.INTER_NEAREST)

        return timestamp, image, depth[..., None], self.mask[..., None], pose



def load_dataset(config, test=False):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(config)
    elif config["Dataset"]["type"] == "scannetpp":
        return ScanNetPP(config, test)
    elif config["Dataset"]["type"] == "azure":
        return AzureKinectDataset(config)
    elif config["Dataset"]["type"] == "kinect_live":
        return AzureKinectLive(config)
    else:
        raise ValueError("Unknown dataset type")