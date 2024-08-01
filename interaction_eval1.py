import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
import importlib
import importlib.util
import io
import os
import sys
import json

import logging
import numpy as np
import torch
import torchvision.models as models
from time import time, sleep
from backbones.shapley import *
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class ScoreFile:
    def __init__(self,score_file_name):
        self.score_file_name=score_file_name
        self.datasets=None
        self.repo_url=None
        self.model=None
        self.pre_options=None
    def set_dataset(self,dataset):
        self.datasets=dataset
    def set_inference(self,repo_url):
        self.repo_url=repo_url
    def set_model(self,model):
        self.model=model
    def set_pre_options(self,pre_opts):
        self.pre_options=pre_opts
    def add_scores(self,atom_id, a_dict):
        return 
    def to_dict(self):
        return
@lru_cache(maxsize=1)
def get_timer():
    return defaultdict(float)
class Atom:
    def __init__(self,atomid,label):
        self.atomid=atomid
        self.label=label

@contextmanager
def timeit(key: str):
    timer = get_timer()
    start_time = time()
    yield
    timer[key] += time() - start_time
def get_net(model_path, eval_dir=None):

    # 动态import model文件
    if not eval_dir:
        eval_dir = os.getcwd()
    # spec = importlib.util.spec_from_file_location(
    #     "module.name", os.path.join(eval_dir, 'model.py')
    # )
    spec = importlib.util.spec_from_file_location("module.name","/home/workspace1/cyy/config/wj.dsc.2022.06.23.res18.FF++.base.analysis/model.py")
    model = importlib.util.module_from_spec(spec)
    sys.path.append(eval_dir)

    spec.loader.exec_module(model)
    print(model_path)
    # checkpoint = io.load(model_path, dict)
    net = model.get()
    # print(net)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        net = net.cuda()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.eval()
    # del checkpoint
    return net

def read_score_file(score_file_path, inference_params_only=False):
    # read the score_file_path
    try:
        with open(score_file_path, 'r') as rf:
            buf = json.loads(rf.read())
        if inference_params_only:  # only get the settings related to the inference
            inference_params_dict = {
                "repo_url": "inference", "model_url": "model",
                "score_file_name": "name", "eval_opts": "pre_options",
                "dataset": "datasets"}
            buf = dict([(arg_key, buf[score_key]) for arg_key, score_key in inference_params_dict.items()])
        return buf, None
    except Exception:
        return None, f'Read score file failed: {score_file_path}'
def update_args_with_score_file(args, buf=None):
    if buf:
        nec_arg_list = ["repo_url", "model_url", "score_file_name", "eval_opts"]
        for arg_key in nec_arg_list:
            setattr(args, arg_key, buf[arg_key] if not getattr(args, arg_key) else getattr(args, arg_key))

        # input dateset depends on whether to overwrite the score_file
        if not args.save_to_new_score_file_path:
            args.dataset = ','.join(list(set(buf['dataset'] + args.dataset.split(','))))
    # assert args.dataset is not None
    # assert args.repo_url is not None
    # assert args.model_url is not None
    # assert args.score_file_name is not None

    print('======================== Current Arguments ========================')
    for (k, v) in vars(args).items():
        print(f"{k}:{v}")
    print('======================== Current Arguments ========================')
    return args
def get_model_sf_name(model_path, inference):
    model_path_splits = model_path.split('/')
    model_name = model_path_splits[-3]
    model_epoch = model_path_splits[-1]
    inference_name = os.path.splitext(inference.split('/')[-1])[0]
    score_file_path = model_path_splits[:-2] + ['eval', model_epoch + '.' + inference_name + '.json']
    return model_name + '/' + model_epoch, '/'.join(score_file_path)
def run(
        score_file: ScoreFile, batch_size=16, threads=8,
        device='cpu0', cache=False, mode="single_frame", context_num=13
):
    model_path = score_file.model
    pre_options = score_file.pre_options
    net = get_net(model_path)
    working = True
    pre_process_err_atom = 0
    while working:
        if mode == "single_frame":
            with timeit('gen_batch_list'):
                batch_aid, batch_label, bad_ld_idxs = [], [], []
                batch_datas = defaultdict(list)
                for _ in range(batch_size):
                    try:
                        # import pdb;pdb.set_trace()
                        # atom_tuple = next(todo_atoms_iter)
                        a_dict={'data':[1,2,3,4]}
                        atom_tuple=[Atom(12,1),a_dict]
                        if atom_tuple[1] is None:
                            pass
                        else:
                            batch_aid.append(atom_tuple[0].atomid)
                            batch_label.append(atom_tuple[0].label)
                            for i, (k, v) in enumerate(atom_tuple[1].items()):
                                batch_datas[k].append(v)          # {'data': [d1, d2, d3, ...]}
                                if np.sum(v) == 0:
                                    bad_ld_idxs.append(_)
                                    pre_process_err_atom += 1
                        if _ == 0:
                            working = False
                            break    
                    except StopIteration:
                        working = False
                        break
            with timeit('gen_bench'):
                if len(batch_datas) == 0:
                    break
                batch_input_datas = {}
                for batch_key, batch_data in batch_datas.items():
                    if len(batch_data) < batch_size:
                        # padding zeros to batch data.
                        batch_input_data = np.array(batch_data)
                        batch_input_data = np.pad(
                            batch_input_data, (
                                (0, batch_size - len(batch_input_data)),
                                *[(0, 0) for _ in range(batch_input_data.ndim - 1)]
                            ), 'edge')
                    else:
                        batch_input_data = np.array(batch_data)
                    batch_input_datas[batch_key] = batch_input_data
            with timeit('gen_score'):
                inputs = torch.Tensor(batch_input_datas['data'])
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                with torch.no_grad():
                    print(batch_label)
                    baseline_imgs = torch.Tensor(np.array(cv2.imread("/home/workspace1/cyy/config/wj.dsc.2022.06.23.res18.FF++.base.analysis/real_fig1/FF_plus-32-frame-bmk_lixi_favorite_Deepfake_FaceForensics_plusplus_Raw_Original_FF++_Deepfakes_661.mp4.png")).transpose(2, 0, 1))
                    imgs = baseline_imgs.reshape(1,3,224,224)
                    phis = shapley_function_visual(imgs, net, info="a", label=batch_label, context_num=context_num)
                    print(phis)


            with timeit('save_score'):
                for i, (atom_id, label, shapley) in enumerate(zip(batch_aid, batch_label, phis)):
                    if i in bad_ld_idxs:
                        # dsc update for debug 20210202
                        continue
                    print(shapley.numpy().tolist())
                    score_file.add_scores(atom_id, {'label': label, 'shapley': shapley.numpy().tolist()})
    print(f'The number of atom failed during pre-process data: {pre_process_err_atom}')
    # print(f'Mean pixel of celebv2: {torch.mean(mean_pixel, dim=(1, 2))/16565}')
    logger.info(get_timer())
    return score_file        

def _main( repo_url, dataset, model, pre_opts,
    score_file_name=None, score_file_path=None, cache=False,
    auc_start=0, auc_end=100.0, 
    mode="single_frame", threads=8, context_num=13):
     # model should be .link file when training, used to locate .link eval file
    if model.endswith('.link'):
        with open(model) as f:
            model_path = f.read().strip()
    else:
        model_path = model
    eval_home = None
    # score_file_name is None when training
    # score_file_name=None
    if score_file_name is None:
        score_file_name, score_file_path = get_model_sf_name(model_path, repo_url)
        score_file_name="aaaaa"
        print(score_file_name)
        if model.endswith('.link'):
            eval_home = os.path.dirname(os.path.dirname(model))
            eval_home = os.path.join(eval_home, 'eval')
    score_file = ScoreFile(score_file_name)
    score_file.set_dataset(dataset)
    score_file.set_inference(repo_url)
    score_file.set_model(model_path)
    score_file.set_pre_options(pre_opts)
    run(score_file, cache=cache, threads=threads, mode=mode, context_num=context_num,batch_size=2)
    results = json.dumps(score_file.to_dict())
    with open(score_file_path, 'w') as wf:
        wf.write(results)
    return
def cli():
    logger.info(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="FF_plus-32-frame-bmk_lixi_favorite_Deepfake_FaceForensics_plusplus_Raw_Manipulation_FF++_NeuralTextures",
                        help='input your dataset name!')
    parser.add_argument('-url', '--repo_url', type=str, default="/home/workspace1/cyy/config/wj.dsc.2022.06.23.res18.FF++.base.analysis/det_encoder_DF_res18base_baselineMean_000n.json",
                        help='input your repo url or omit it by giving the path of an existing score file!')
    parser.add_argument('-m', '--model_url', type=str, default="/home/workspace1/cyy/config/models/models/res18_base_epoch_22.pt",
                        help='input your model url or omit it by fiving the path of  an existing score file!')
    # parser.add_argument('-n', '--score_file_name', type=str, help='input your score_file name!', default="DF_det_encoder_res18")
    parser.add_argument('-n', '--score_file_name', type=str, help='input your score_file name!', default=None)
    # parser.add_argument('-f', '--score_file_path', type=str, help='input your score_file path!', default="/home/workspace1/cyy/config/jsonfiles/shapley_multiframe/det_encoder_DF_res18aug_baselineMean_000n.json")
    parser.add_argument('-f', '--score_file_path', type=str, help='input your score_file path!', default=None)
    parser.add_argument('--start', action='store_true', default=False, help='new process')
    parser.add_argument('--mode', default='single_frame', help='eval mode single frame or multi frame eval.')
    parser.add_argument('--eval-opts', default='{}', type=json.loads)
    parser.add_argument('--eval-cache', action="store_true", help='set to use eval cache')
    parser.add_argument('--auc-start', type=float, default=0.)
    parser.add_argument('--auc-end', type=float, default=100.0)
    # parser.add_argument('--username', type=str, default=DEFAULT_USERNAME)
    parser.add_argument('-o', '--save_to_new_score_file_path', type=str, default=None,
                        help='input a new score_file_path to save results, instead of adding to original score_file_path.')
    parser.add_argument('--threads', type=int, default=len(os.sched_getaffinity(0)) * 2)
    parser.add_argument('--context_num', type=int, default=13)
    args = parser.parse_args()

    buf, error_msg = read_score_file(args.score_file_path, inference_params_only=True)
    if error_msg:
        logger.warning(error_msg)
    args = update_args_with_score_file(args, buf)

    # overwrite score_file
    if args.save_to_new_score_file_path:
        args.score_file_path = args.save_to_new_score_file_path

    if not args.start:
        
        _main(args.repo_url, args.dataset.split(','), args.model_url, args.eval_opts,
              score_file_name=args.score_file_name, score_file_path=args.score_file_path,
              cache=args.eval_cache, auc_start=args.auc_start, auc_end=args.auc_end,
              mode=args.mode, threads=args.threads, context_num=args.context_num)
        return 0
    return

if __name__ == '__main__':
    sys.exit(cli())