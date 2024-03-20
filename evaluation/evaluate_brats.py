import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import numpy as np
import cv2
import sys
import argparse
import json
import torch
import torch_fidelity


def eval_brats(method, real_path, fake_path, seg_save_path):
    # Download pre-trained tumor segmentation model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
                            in_channels=3, out_channels=1, init_features=32, pretrained=True)

    reals = glob.glob(real_path + '/*.png')
    fakes = glob.glob(fake_path + '/*.png')

    reals = sorted(reals)
    fakes = sorted(fakes)
    print(real_path, fake_path)

    num_imgs = len(reals)
    pix_count = 0.0
    RMSE = 0.0
    FP_pix_count = 0.0
    FP_inst_count = 0.0
    
    if not os.path.exists(seg_save_path):
        os.makedirs(seg_save_path)

    print("number of testing images :", num_imgs)
    model.eval()
    for i in range(num_imgs):
        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])
        real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Tumor segmentation
        fake_tensor = np.transpose(fake, (2, 0, 1))
        fake_tensor = torch.from_numpy(fake_tensor/255).float().unsqueeze(0)
        with torch.no_grad():
            tumor_segmentation = model(fake_tensor)
        tumor_segmentation = (tumor_segmentation[0][0].detach().cpu().numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(seg_save_path, os.path.basename(fakes[i])), tumor_segmentation)

        # Calculate false positive rate of tumors
        FP = (tumor_segmentation/255 > 0.5)
        FP_pix_count = FP_pix_count + np.sum(FP)
        pix_count = pix_count + tumor_segmentation.shape[0]*tumor_segmentation.shape[1]
        if np.sum(FP)>= 50:
            FP_inst_count = FP_inst_count + 1
            print("false positive at:", i)

        real = real.astype(np.float32)
        fake = fake.astype(np.float32)
        diff = np.abs(real - fake)


        diff = (diff**2)/(256**2)
        diff = np.sum(diff)
        rmse = np.sqrt(diff)
        RMSE = RMSE + rmse
    
    pFPR100 = FP_pix_count/pix_count*100*100
    iFPR = FP_inst_count/num_imgs*100

    RMSE = RMSE/num_imgs
    
    eval_dict = {'method': method, 'RMSE':RMSE, 'pFPR%d'%(100):pFPR100, 'iFPR':iFPR}

    eval_args = {'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=real_path, input2=fake_path, **eval_args)
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    print('[*] evaluation finished!')

    
    print('rmse: {0}, pFPR: {1}, iFPR: {2}'.format(RMSE, pFPR100, iFPR))
    return eval_dict


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_path', type=str, help='path to the ground truth images')
    parser.add_argument('--pred_path', type=str, help='path to the generated images')
    parser.add_argument('--output_path', type=str, help='path to save the evaluation results')
    parser.add_argument('--seg_save_path', type=str, help='path to save the tumor segmentation results')
    parser.add_argument('--dataset', type=str, default='Google')
    parser.add_argument('--method', type=str, default='StegoGAN')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    return args

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_metrics = eval_brats(args.method, args.gt_path, args.pred_path, args.seg_save_path)
    with open(os.path.join(args.output_path, args.dataset + '_result.json'), 'w') as fp:
        json.dump(eval_metrics, fp)


if __name__ == '__main__':
    main()