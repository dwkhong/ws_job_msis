import os
import contextlib
import argparse
import numpy as np
import torch
import checkmAP_main
import warnings
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)
warnings.filterwarnings('ignore')
# sys.path.insert(0, './')
torch.manual_seed(0)
np.random.seed(0)

@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)

def parse_args():

    parser = argparse.ArgumentParser('')
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--vis', dest='vis',
                        default=False, type=bool)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    parser.add_argument('--savePath', default='results')
    parser.add_argument('--imgSize', default='1920,1080')

    args = parser.parse_args()
    return args

def test(args):

    val_dir = "BDD_10K_GT"
    #names = ["Vehicle", "Motorcycle", "Bicycle", "Pedestrian"]
    names = ["Vehicle", "Rider", "Pedestrian"]
    #det_folder = "BDD_10K_1_640_ov_25"
    det_folder = "BDD_10K_PATCHES_RCT"
    #det_folder = "BDD_10K_640"
    #det_folder = "BDD_10K_MMS_01"
    #det_folder = "BDD_10K_2_640"
    #det_folder = "BDD_10K_2_FP"
    
    #if args.dataset == 'custom':
    #    data_dict = check_dataset(args.data)
    #    _, val_data, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
    #    nc = int(data_dict['nc'])  # number of classes
    #    names = data_dict['names']  # class names
    #    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {args.data}'  # check


    args.gtFolder        =  val_dir
    args.detFolder       =  det_folder
    args.iouThreshold    =  0.5
    args.gtFormat        =  'xywh'
    args.detFormat       =  'xywh'
    args.gtCoordinates   =  'rel'
    args.detCoordinates  =  'rel'
    args.imgSize         =  '1280,720'  # for bdd --> 1280, 720 and waymo --> 1920, 1280 ## 1920,1536
    args.savePath        =  'results'
    args.call_with_train =  False
    args.showPlot        =  False
    args.names           =  names
    args.val             =  True
    map, class_metrics = checkmAP_main.main(args)

    return map, class_metrics   


if __name__ == '__main__':
    args = parse_args()
    map, metrics = test(args)
