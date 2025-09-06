import os, shutil, math, argparse, random, lpips, json
import PIL.Image as Image
from tqdm import tqdm 
import pandas as pd
import torch, torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from loss import LossComputer

from io import BytesIO
import pickle

from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.imagenet_dataset import ImageNetDataset

from nic_models import TCM

import torch.nn.functional as F

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def pickle_size_of(obj) -> int:
    buffer = BytesIO()
    pickle.dump(obj, buffer)

    num_bytes = buffer.getbuffer().nbytes

    buffer.close()

    return num_bytes

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def main():
    
    split_dict = {'train': 0, 'val': 1, 'test': 2}
    
    source_path = '/home/minkyu4506/group_DRO_with_NIC_with_DINOv2_pre_trained_NIC'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nic-name', type=str, default='TCM') # TCM or ILLM
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    ##########################################
    # Random 제어 
    random.seed(args.seed)
    np.random.seed(int(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    
    loss_fn_alex = lpips.LPIPS(net='alex').eval()
    loss_fn_alex = loss_fn_alex.to('cuda:0')
    loss_fn_alex.requires_grad_(False)
    
    logger = Logger(f'{source_path}/eval_logger_{args.nic_name}.txt', mode='w')    
    
    dataset_name = 'imagenet_ball'
    target_class_index_list = []
    
    with open('/home/minkyu4506/Check_ImageNet_with_NIC/cls_idx_dict_ball.json', 'r') as f:
        cls_idx_dict = json.load(f)
    
    for cls_name_with_idx in cls_idx_dict:
        cls_idx = cls_name_with_idx[0].split('_')[-1]
        target_class_index_list.append(int(cls_idx))
    
    with open('/home/minkyu4506/Check_ImageNet_with_NIC/data/imagenet_class_index.json', 'r') as f:
        imagenet_class_index = json.load(f)
        
    cls_id_list = []
    
    for cls_idx in imagenet_class_index:
        cls_id, cls_name = imagenet_class_index[cls_idx]
        if int(cls_idx) in target_class_index_list:
            cls_id_list.append(cls_id)
    
    # valid set
    filename_dict = {}
    with open('/home/minkyu4506/Check_ImageNet_with_NIC/data/imagenet_valid_image_list_per_class.json', 'r') as f:
        imagenet_valid_image_list_per_class = json.load(f)
    
    for cls_idx, cls_id in enumerate(cls_id_list):
        
        filename_dict[cls_id] = []
            
        image_list_per_clss = imagenet_valid_image_list_per_class[cls_id]

        for image_name in image_list_per_clss:
            filename_dict[cls_id].append(f'/home/minkyu4506/Fairness_datasets/imagenet_validset/{image_name}')

    transform_images = torchvision.transforms.ToTensor()
    
    for image_quality in range(1, 7):
        
        log_path = f'{source_path}/results_logs/nic_{args.nic_name}/datset_{dataset_name}/iq_{image_quality}'

        if os.path.exists(log_path) == False:
            try:
                os.mkdir(log_path)
            except:
                os.makedirs(log_path)    
        
        image_save_folder_path = f'{source_path}/compress_images_pretrained_models/nic_{args.nic_name}/datset_{dataset_name}/iq_{image_quality}'
        save_path_for_visualization = f'{source_path}/compress_images_pretrained_models_for_visualization/nic_{args.nic_name}/datset_{dataset_name}/iq_{image_quality}'
        
        if os.path.exists(image_save_folder_path) == True:
            shutil.rmtree(image_save_folder_path)
        
        if os.path.exists(save_path_for_visualization) == True:
            shutil.rmtree(save_path_for_visualization)
            
        try:
            os.mkdir(image_save_folder_path)
            os.mkdir(save_path_for_visualization)
        except:
            os.makedirs(image_save_folder_path)
            os.makedirs(save_path_for_visualization)
        
    
        if args.nic_name == 'TCM':
            nic_eval = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
            checkpoint = torch.load(f'{source_path}/nic_models/checkpoints/TCM_pretrained_image_quality_{image_quality}.pth.tar', map_location = 'cpu')  
            
            net_state_dice = nic_eval.state_dict()    
            try:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
            except:
                new_state_dict = {}
                for k, v in checkpoint.items():
                    new_state_dict[k.replace("module.", "")] = v

            # 원래 있던 애들(?) 만 따로 사용
            for k, v in new_state_dict.items():
                net_state_dice[k] = new_state_dict[k]

            nic_eval.load_state_dict(net_state_dice) 
            
        elif args.nic_name == 'ILLM':
            nic_eval = torch.hub.load("facebookresearch/NeuralCompression", f'msillm_quality_{image_quality}')
        
        nic_eval.eval()
        nic_eval.requires_grad_(False)
        nic_eval.update()

        statistics = {}
        
        count = 0
        bpp_mean = 0
        psnr_mean = 0
        lpips_mean = 0
        
        # logger = Logger(os.path.join(log_path, 'log.txt'), 'w')
        
        for cls_id in filename_dict:
            
            statistics[cls_id] = {}
            
            bpp_list = []
            psnr_list = []
            lpips_list = []
            
            for idx, file_path in enumerate(filename_dict[cls_id]):
                
                    
                img = Image.open(file_path).convert('RGB')
                x = transform_images(img).unsqueeze(0)
                
                num_pixels = x.size(-2) * x.size(-1)
                
                if num_pixels > (1920 * 1080):
                    device = 'cpu'
                    force_cpu = True
                else:
                    device = 'cuda:0'
                    force_cpu = False
                
                x = x.to(device)
                nic_eval = nic_eval.to(device)
                
                if (args.nic_name == 'ILLM') and (device == 'cuda:0'):
                    nic_eval.update_tensor_devices("compress")
                    
    
                if args.nic_name == 'TCM':
                    x_padded, padding = pad(x, 256)
                    
                    x_padded = x_padded.to(x.device)
                    
                    out_enc = nic_eval.compress(x_padded)
                    out_dec = nic_eval.decompress(out_enc["strings"], out_enc["shape"])
                    
                    x_hat = out_dec["x_hat"]
                    x_hat = crop(x_hat, padding)
                    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                
                elif args.nic_name == 'ILLM':
                    compressed = nic_eval.compress(x, force_cpu=force_cpu)
            
                    num_bytes = pickle_size_of(compressed)
                    bpp = num_bytes * 8 / num_pixels
                    x_hat = nic_eval.decompress(compressed, force_cpu=force_cpu).clamp(0.0, 1.0)
                
                image_name = file_path.split('/')[-1]
                
                output_path = f'{image_save_folder_path}/{image_name}'
                torchvision.utils.save_image(x_hat, output_path, nrow=1) 
                
                # psnr, lpips 측정하기 
                loss_fn_alex = loss_fn_alex.to(x.device)
                
                psnr = compute_psnr(x, x_hat)
                lpips_score = loss_fn_alex(x, x_hat).item()


                logger.write(f'Dataset: {dataset_name}, cls_id: {cls_id}, image quality: {image_quality}, Image: {image_name}, PSNR: {psnr:.3f}, LPIPS: {lpips_score:.3f}\n') 
                logger.flush()
                
                bpp_list.append(bpp)
                psnr_list.append(psnr)
                lpips_list.append(lpips_score)
                
                bpp_mean += bpp
                psnr_mean += psnr
                lpips_mean += lpips_score
                count += 1
                
                if idx % 20 == 0:
                    torchvision.utils.save_image(x_hat, f'{save_path_for_visualization}/{image_name}', nrow=1) 
                
            statistics[cls_id]['bpp'] = torch.tensor(bpp_list).mean().item()
            statistics[cls_id]['psnr'] = torch.tensor(psnr_list).mean().item()
            statistics[cls_id]['lpips'] = torch.tensor(lpips_list).mean().item()
            
            logger.write(f'Group: {cls_id}, Mean bpp: {statistics[cls_id]["bpp"]:.3f}, PSNR: {statistics[cls_id]["psnr"]:.3f}, LPIPS: {statistics[cls_id]["lpips"]:.3f}\n')
            
            
        bpp_mean /= count
        psnr_mean /= count
        lpips_mean /= count
        
        statistics['total'] = {
            'bpp': bpp_mean,
            'psnr': psnr_mean,
            'lpips': lpips_mean
        }
        
        with open(f'{log_path}/rd_statistics.json', 'w') as f:
            json.dump(statistics, f, indent=4)
        
        logger.write(f'Total Mean bpp: {bpp_mean:.3f}, PSNR: {psnr_mean:.3f}, LPIPS: {lpips_mean:.3f}\n')
        logger.flush()
        
        full_dataset = ImageNetDataset(
            train_dir='/public-dataset/ILSVRC2012',
            val_dir = image_save_folder_path,
            model_type='resnet50',
            mask_dir = 'None', target_class_index_list= target_class_index_list)

        subsets = full_dataset.get_splits(['val'], train_frac=1.0)
        
        test_data = DRODataset(subsets['val'], full_dataset.y_array, full_dataset.group_array,
                               process_item_fn=None, n_groups=full_dataset.n_groups,
                                n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str)

        loader_kwargs = {'batch_size':8, 'num_workers':4, 'pin_memory':True}
            
        for cls_model_type in ['erm', 'dro']:
            
            stat_per_group = {}
            
            test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
            
            test_loss_computer = LossComputer(
                    criterion_train = torch.nn.CrossEntropyLoss(reduction='none'),
                    is_robust=False,
                    dataset=test_data,
                    step_size=0.01,
                    alpha=0.01,
                    gamma=0.1,
                    is_train = False)
            
            cls_model = torchvision.models.resnet50(weights=None)
            cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, len(cls_idx_dict))
            
            if cls_model_type == 'dro':
                cls_model_ckpt = torch.load('/home/minkyu4506/Check_ImageNet_with_NIC/experiments_ball/img_net_specific_classes_is_robust_True/last_model.pth', map_location = 'cpu')['state_dict']
            elif cls_model_type == 'erm':
                cls_model_ckpt = torch.load('/home/minkyu4506/Check_ImageNet_with_NIC/experiments_ball/img_net_specific_classes_is_robust_False/last_model.pth', map_location = 'cpu')['state_dict']
            
            cls_model.load_state_dict(cls_model_ckpt)
        
            cls_model = cls_model.to('cuda:0')
            cls_model.eval()
            cls_model.requires_grad_(False)
            
            test_csv_logger = CSVBatchLogger(os.path.join(log_path, f'{cls_model_type}_performance_results.csv'), test_data.n_groups, mode='w')
            
            for batch_idx, batch in enumerate(test_loader):
                
                batch = tuple(t.cuda() for t in batch)
                x = batch[0]
                y = batch[1]
                g = batch[2]
                
                outputs = cls_model(x)

                test_loss_computer.loss(nic_outputs = None, x = outputs, y = y, group_idx = g)

            test_csv_logger.log(1, batch_idx, test_loss_computer.get_stats())
            test_csv_logger.flush()
            
            for group_idx in range(test_loss_computer.n_groups):
                
                temp_key = f'avg_acc_group:{group_idx}'
                
                stat_per_group[temp_key] = test_loss_computer.avg_group_acc[group_idx].item()
            
            stat_per_group['avg_acc'] = test_loss_computer.avg_acc.item()
            
            with open(f'{log_path}/{cls_model_type}_performance_result.json', 'w') as f:
                json.dump(stat_per_group, f, indent=4)
            
        # 용량 아끼기 위해 지운다
        shutil.rmtree(image_save_folder_path)
        
if __name__=='__main__':
    main()