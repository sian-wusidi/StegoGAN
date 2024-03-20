import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import numpy as np
import cv2
import sys
import argparse
import json
import torch_fidelity

"""
This code is based on https://github.com/Mid-Push/Decent/
"""

def eval_google(method, real_path, fake_path, thr1=5, thr2=10):
    reals = glob.glob(real_path + '/*.png')
    fakes = glob.glob(fake_path + '/*.png')

    reals = sorted(reals)
    fakes = sorted(fakes)
    print(real_path, fake_path)

    num_imgs = len(fakes)
    corr5_count = 0.0
    corr10_count = 0.0
    pix_count = 0.0
    RMSE = 0.0
    FP_pix_count = 0.0
    FP_inst_count = 0.0
    
    color_protype = np.array([[[30, 160, 240]]])  # 1, 1, 3, BGR

    print("number of testing images :", num_imgs)
    for i in range(num_imgs):

        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])
        import pdb
        if real is None or fake is None:
            print("Error: real or fake is None")
            pdb.set_trace()
        real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)

        real = real.astype(np.float32)
        fake = fake.astype(np.float32)
        diff = np.abs(real - fake)

        max_diff = np.max(diff, axis=2)

        corr5_count = corr5_count + np.sum(max_diff < thr1)
        corr10_count = corr10_count + np.sum(max_diff < thr2)
        pix_count = pix_count + 256**2

        FP_diff = np.abs(fake - color_protype)
        max_FP_diff = np.max(FP_diff, axis=2)
        FP = max_FP_diff < 20
        FP_pix_count = FP_pix_count + np.sum(FP)
        if np.sum(FP):
            FP_inst_count = FP_inst_count + 1
            print("false positive at:", fakes[i])

        diff = (diff**2)/(256**2)
        diff = np.sum(diff)
        rmse = np.sqrt(diff)
        RMSE = RMSE + rmse

    pFPR100 = FP_pix_count/pix_count*100*100
    iFPR = FP_inst_count/num_imgs*100

    RMSE = RMSE/num_imgs
    acc5 = corr5_count/pix_count*100.
    acc10 = corr10_count/pix_count*100.
    eval_dict = { 'method': method, 'RMSE':RMSE,'acc%d'%(thr1):acc5, 'acc%d'%(thr2):acc10, 'pFPR%d'%(100):pFPR100, 'iFPR':iFPR}

    eval_args = {'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=real_path, input2=fake_path, **eval_args)
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    print('[*] evaluation finished!')
    
    print('rmse: {0}, acc5: {1}, acc10: {2}, pFPR: {3}, iFPR: {4}'.format(RMSE, acc5, acc10, pFPR100, iFPR))
    return eval_dict


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_path', type=str, help='path to the ground truth images')
    parser.add_argument('--pred_path', type=str, help='path to the generated images')
    parser.add_argument('--output_path', type=str, help='path to save the evaluation results')
    parser.add_argument('--dataset', type=str, default='Google')
    parser.add_argument('--method', type=str, default='StegoGAN')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    return args

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    eval_metrics = eval_google(args.method, args.gt_path, args.pred_path)
    with open(os.path.join(args.output_path, args.dataset + '_result.json'), 'w') as fp:
        json.dump(eval_metrics, fp)
    

if __name__ == '__main__':
    main()