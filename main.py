import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submodules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/utils'))
from argparse import ArgumentParser
from datetime import datetime
import time
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from src.utils.dataset import load_dataset
from src.utils.frame import Frame
from src.system import EGGFusion

def load_config(path):
    scene_config = OmegaConf.load(path)
    data_config = OmegaConf.load(scene_config.data_config)
    base_config = OmegaConf.load(scene_config.base_config)
    cfg = OmegaConf.merge(base_config, data_config, scene_config)

    # create workspace
    root_dir    = cfg.System.root_dir
    dataset     = cfg.Dataset.type
    scene       = cfg.Dataset.scene
    timestamp   = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_dir    = dataset + '_' + scene + '_' + timestamp

    cfg.System.save_dir = os.path.join(root_dir, save_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(cfg.System.save_dir):
        os.makedirs(cfg.System.save_dir)

    with open(os.path.join(cfg.System.save_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)

    return cfg

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    mp.set_start_method("spawn")
    
    config = load_config(args.config)
        
    dataset = load_dataset(config=config)

    ef = EGGFusion(config)

    for fid in range(len(dataset)):
        
        print(f"Processing frame {fid}/{len(dataset)}")

        curr_frame = Frame.init_from_dataset(dataset, fid, config.Dataset.preload)
        
        ef.reconstruct(curr_frame)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    ef.finish()

    if config.System.eval_tracking:
        ef.evaluate_trajectory()