#from train_new import train
import argparse
import os
import pickle
import torch
from .utils.google_utils import attempt_download
from .models.experimental import attempt_load

import yaml
from .models.yolo import Model
from .utils.torch_utils import intersect_dicts
import logging
from .utils.datasets import externalMemory
import torch.nn as nn
import torch.optim as optim
# from utils.general import (
#     torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
#     compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
#     check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)

from .utils.general import check_dataset, check_file, init_seeds, increment_dir

from warnings import warn
from pathlib import Path
from .train import train_on_large_batch



from .detect import detect_img
from .fix_class_id import fix_class_id


model_cfg = 'models/yolov5s.yaml'


def load_state(path='detector.pckl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Detector:

    def __init__(self, load_ext_mem=False):
        self.logger = logging.getLogger(__name__)

        self.device = torch.device('cuda:0')


        opt, hyp = get_opt_and_hyp()

        self.logger.info(f'Hyperparameters {hyp}')
        # self.log_dir = Path(opt.logdir) # logging directory
        self.log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1
        self.log_dir = Path(self.log_dir)
        wdir = self.log_dir / 'weights'  # weights dfirectory
        os.makedirs(wdir, exist_ok=True)
        # last = wdir / 'last.pt'
        # best = wdir / 'best.pt'
        # results_file = str(log_dir / 'results.txt')
        # epochs, batch_size, total_batch_size, weights, rank = \
        #     opt.epochs_init, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

        # Save run settings
        with open(self.log_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        with open(self.log_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

        # Configure
        init_seeds(2)
        # with open(opt.data) as f:
        #     data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
 
       # check_dataset(data_dict)  # check


        self.nc = 6
        self.names = ['???'] * 6 
       # assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

        # Model
        weights = opt.weights
        pretrained = weights.endswith('.pt')
        if pretrained:
            attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
            if hyp.get('anchors'):
                ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
            self.model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc).to(self.device)  # create
            exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), weights))  # report
        else:
            self.model = Model(opt.cfg, ch=3, nc=self.nc).to(self.device)  # create

        # Freeze
        # freeze = []  # parameter names to freeze (full or partial)
        # for k, v in self.model.named_parameters():
        #     v.requires_grad = True  # train all layers
        #     if any(x in k for x in freeze):
        #         print('freezing %s' % k)
        #         v.requires_grad = False


        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / opt.total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= opt.total_batch_size * accumulate / nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        self.optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        #logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        # lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
        #self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        
        # Resume
        # start_epoch = 0
        self.best_fitness = 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # Epochs
            # start_epoch = ckpt['epoch'] + 1


            del ckpt, state_dict

        # Image sizes
        self.gs = int(max(self.model.stride))  # grid size (max stride)
        


        # Model parameters
        hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)


        # extenral memory
        self.extMem = externalMemory(size=200)
        print(f'external memory file: {self.extMem.get_memory_file()}')
        self.opt = opt
        self.hyp = hyp

        self.added_classes = 0
      #  self.train.n_iter = 0

    def train(self, train_files, class_names, valid_file=None):

        
       # self.log_dir = Path(str(self.log_dir) + str(train.n_iter))

        if not isinstance(class_names, list):
            class_names = [class_names]
        n_classes_to_add = len(class_names)

        if not isinstance(train_files, list):
            train_files = [train_files]

        fix_class_id(train_files, class_names, self.added_classes)
        # fix_class_id([valid_file], class_names, self.added_classes)

        use_ext_mem = False if self.added_classes == 0 else True
        for cn in class_names:
            self.names[self.added_classes] = cn
            self.added_classes += 1


        


        train_on_large_batch(n_classes_to_add, train_files, self.model, self.device, self.logger,
                             valid_path=valid_file, imgsz=self.opt.img_size,
                             imgsz_test=self.opt.img_size, gs=self.gs,
                             opt=self.opt, hyp=self.hyp, nc=self.nc, log_dir=self.log_dir, tb_writer=None,
                             names=self.names, optimizer=self.optimizer,
                             extMem=self.extMem, best_fitness=self.best_fitness, use_ext_mem=use_ext_mem)


        #self.model = attempt_load('runs/weights/best.pt')

    def save_state(self, path='detector.pckl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def detect(self, image):
        # image: array
        # return [new_image, bboxes]
        return detect_img(image, self.model, augment=True)

    def find_object(self, object_name:str, image):
        # returns x,y,w,h of found object
        # if unknown object return "unknown object"
        # if not found return "not found"
        det = detect_img(image, self.model)[1]

        if object_name not in self.names:
            print(f'Unknown object:{object_name} in list: {self.names}')
            return

        obj_index = self.names.index(object_name)
        ret_bboxes = det[det[..., -1] == obj_index]

        if ret_bboxes.nelement() == 0:
            print(f'{object_name} not found')
            return

        return ret_bboxes  # each bbox: [x1, y1, x2, y2, confidence, class_index]


    # def detect_from_web_cam():


def get_opt_and_hyp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='../.dddd./../demoset/data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=15)
    # parser.add_argument('--epochs_iter', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=416, help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--log-imgs', type=int, default=10, help='number of images for W&B logging, max 100')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    # opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    # assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'



    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    return opt, hyp

if __name__ == '__main__':

    dt = Detector()

    train_paths = ['dataset_files/Cube.txt',
                    'dataset_files/Banana.txt',
                    'dataset_files/Box.txt',
       #             'dataset_files/Toy.txt',
    ]
    class_names = ['Cube', 
                'Banana', 
                'Box', 
            #    'Toy'
                ]
    dt.train(train_paths, class_names)#, valid_file='valid.txt')

    dt.save_state('main.pckl')

    train_paths = ['dataset_files/Can.txt'
           ]
    class_names = ['Can']


    dt = load_state('main.pckl')
    dt.train(train_paths, class_names)#, valid_file='valid.txt')
    dt.save_state('iter_4.pckl')
    train_paths = [
                    'dataset_files/Egg.txt'
           ]
    class_names = ['Egg']


    dt = load_state('iter_4.pckl')
    dt.train(train_paths, class_names)#, valid_file='valid.txt')
    dt.save_state('iter_5.pckl')
    # dt = Detector()

    # train_paths = ['dataset_files/Cube.txt',
    #                 'dataset_files/Banana.txt',
    #                 'dataset_files/Box.txt',
    #                 'dataset_files/Toy.txt',
    #                 'dataset_files/Can.txt',
    #                 'dataset_files/Egg.txt'
    # ]

    # class_names = ['Cube', 'Banana', 'Box', 'Toy', 'Can', 'Egg']
    # dt.train(train_paths, class_names, valid_file='valid.txt')

    # dt.save_state('all_classes.pckl')




