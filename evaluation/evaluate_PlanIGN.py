import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import glob
import numpy as np
import cv2
import sys
import argparse
import json
import json
from scipy import ndimage
from sklearn.metrics import jaccard_score, precision_score, recall_score
import torch_fidelity

def eval_IGN(method, real_path, real_path_topo, fake_path, fake_path_mask, thr1=2, thr2=5):
    reals = glob.glob(real_path + '/*.png')
    fakes = glob.glob(fake_path + '/*.png')
    reals_mask = glob.glob(real_path_topo + '/*.png')
    fakes_mask = glob.glob(fake_path_mask + '/*.png')

    reals = sorted(reals)
    fakes = sorted(fakes)
    reals_mask = sorted(reals_mask)
    fakes_mask = sorted(fakes_mask)
    print(real_path, real_path_topo, fake_path, fake_path_mask)

    num_imgs = len(fakes)
    corr5_count = 0.0
    corr10_count = 0.0
    pix_count = 0.0
    RMSE = 0.0
    MP = 0.0
    MR = 0.0
    MIOU = 0.0
    
    print("number of testing images :", num_imgs)
    for i in range(num_imgs):

        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])
        real_topo = cv2.imread(reals_mask[i])
        fake_mask = cv2.imread(fakes_mask[i],0)

        real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)
        real_topo = cv2.resize(real_topo, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake_mask = cv2.resize(fake_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        real = real.astype(np.float32)
        fake = fake.astype(np.float32)
        real_topo = real_topo.astype(np.float32)
        fake_mask = fake_mask.astype(np.float32)

        # Get text mask
        mask = np.abs(real_topo - real)
        max_mask = np.max(mask, axis=2)
        real_mask = (max_mask > 0).astype(int)
        fake_mask = (fake_mask > 127.).astype(int)

        # Dilate the GT mask twice
        struct1 = ndimage.generate_binary_structure(2,2)
        real_mask = ndimage.binary_dilation(real_mask, structure=struct1).astype(real_mask.dtype)
        real_mask = ndimage.binary_dilation(real_mask, structure=struct1).astype(real_mask.dtype)  

        # Calculate metrics
        mIoU = jaccard_score(fake_mask, real_mask, average = 'micro')
        mPrecision = precision_score(real_mask, fake_mask, average='micro')
        mRecall = recall_score(real_mask, fake_mask, average='micro')

        diff = np.abs(real - fake)
        max_diff = np.max(diff, axis=2)

        corr5_count = corr5_count + np.sum(max_diff < thr1)
        corr10_count = corr10_count + np.sum(max_diff < thr2)
        pix_count = pix_count + 256**2

        diff = (diff**2)/(256**2)
        diff = np.sum(diff)
        rmse = np.sqrt(diff)
        RMSE = RMSE + rmse
        MP = MP + mPrecision
        MR = MR + mRecall
        MIOU = MIOU + mIoU

    RMSE = RMSE/num_imgs
    MP = MP/num_imgs * 100
    MR = MR/num_imgs * 100
    MIOU = MIOU/num_imgs * 100
    acc5 = corr5_count/pix_count*100.
    acc10 = corr10_count/pix_count*100.
    eval_dict = { 'method': method, 'RMSE':RMSE,'acc%d'%(thr1):acc5, 'acc%d'%(thr2):acc10, 'mIOU(mask)':MIOU, 'precision(mask)': MP, 'recall(mask)': MR}

    eval_args = {'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=real_path, input2=fake_path, **eval_args)
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    print('[*] evaluation finished!')

    
    print('rmse: {0}, acc2: {1}, acc5: {2}'.format(RMSE, acc5, acc10))
    return eval_dict


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_path_TU', type=str, help='path to the ground truth images without toponyms')
    parser.add_argument('--gt_path_T', type=str, help='path to the ground truth images with toponyms')
    parser.add_argument('--pred_path', type=str, help='path to the generated images')
    parser.add_argument('--pred_path_mask', type=str, help='path to the generated mimstach masks')
    parser.add_argument('--output_path', type=str, help='path to save the evaluation results')
    parser.add_argument('--dataset', type=str, default='PlanIGN')
    parser.add_argument('--method', type=str, default='StegoGAN')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    return args

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    eval_metrics = eval_IGN(args.method, args.gt_path_TU, args.gt_path_T, args.pred_path, args.pred_path_mask)
    with open(os.path.join(args.output_path, args.dataset + '_result.json'), 'w') as fp:
        json.dump(eval_metrics, fp)
    

if __name__ == '__main__':
    main()