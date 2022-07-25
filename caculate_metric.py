'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import lpips


def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = './ntire20/track1-valid-gt'
    folder_Gen = './ntire20/track1-valid-input_RRDB'

    f = open('./dataset/ntire20/metrics.txt','a+')
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    crop_border = 4
    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []

    SSIM_all = []

    LPIPS_all = []

    img_list = sorted(glob.glob(folder_GT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        #Test LPIPS
        img0 = lpips.im2tensor(lpips.load_image(img_path))
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(folder_Gen, base_name + suffix + '.png')))
        img0 = img0.cuda()
        img1 = img1.cuda()

        dist01 = loss_fn.forward(img0,img1)

        lp = dist01.cpu().detach().numpy()

        LPIPS_all.append(lp)

        im_GT = cv2.imread(img_path) / 255.
        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.     
        
        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)

        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tLPIPS: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM, lp[0][0][0][0]))
        f.writelines('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tLPIPS: {:.6f}\n'.format(
            i + 1, base_name, PSNR, SSIM, lp[0][0][0][0]))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    avg_lpips = sum(LPIPS_all)/ len(LPIPS_all)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, LPIPS: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all),
        float(avg_lpips[0][0][0][0])))
    f.writelines('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, LPIPS: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all),
        avg_lpips[0][0][0][0]))

    f.close()
        


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

if __name__ == '__main__':
    main()