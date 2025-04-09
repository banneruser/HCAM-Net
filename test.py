from astropy.io import fits
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
from models.STN import register_model
from torchvision import transforms
from data import dataset, trans
from models.Unet import DH_net
import matplotlib.pyplot as plt
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter
from torch.utils.data._utils.collate import default_collate
from time import time


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(512, 512)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1] = 1  # 在列方向上设置网格线
    for i in range(0, grid_img.shape[0], grid_step):
        grid_img[i+line_thickness-1, :] = 1  # 在行方向上设置网格线
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).float()  # 转换为 float 类型
    return grid_img


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a 2D displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D displacement field of size (2, H, W)
    Returns:
        jacobian determinant (H x W array)
    """

    volshape = disp.shape[-2:]  # Get the spatial shape (H, W)

    # Compute grid
    grid_lst = np.meshgrid(np.arange(volshape[1]), np.arange(volshape[0]))
    grid = np.stack(grid_lst, axis=2)

    # Compute gradients
    J = np.gradient(disp, axis=(1, 2))

    dfdx = J[0]
    dfdy = J[1]

    return dfdx[0] * dfdy[1] - dfdy[0] * dfdx[1]



def main():
    device = torch.device(GPU_iden if torch.cuda.is_available() else "cpu")
    test_dir = 'E:/data/pre/fits_data/'
    save_path = 'E:/data/pre/Net-net/1225NewNet/12/'

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # if not os.path.exists(save_path1):
    #     os.makedirs(save_path1)

    img_size = (512, 512)

    model = DH_net()

    model.load_state_dict(torch.load('experiments/0.005NewNET500/model_epoch_300.pth'), strict=False)

    model.cuda()
    model.eval()

    reg_model = register_model(img_size, 'bilinear')
    reg_model.cuda()

    test_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
    ])
    test_set = dataset.testReadDataset(test_dir, transforms=test_composed)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    eval_det = AverageMeter()

    total_time = 0  # 总时间
    num_pairs = 0  # 图像对计数

    i = 1
    with torch.no_grad():

        for step, data in enumerate(tqdm(test_loader)):
            start_time = time()
            model.eval()

            x = data[0].to(device)
            y = data[1].to(device)

            # fixed_max = data[2].cpu().numpy()
            # fixed_min = data[3].cpu().numpy()
            moving_max = data[4].cpu().numpy()
            moving_min = data[5].cpu().numpy()

            moving_filename = os.path.splitext(data[7][0])[0]


            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            jac_det = jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())

            grid_img = mk_grid_img(20, 1, (512, 512))

            def_flow = reg_model([grid_img.cuda().float(), flow.cuda()])

            def_flow = def_flow.squeeze().cpu().numpy()

            x_cpu=x_def.squeeze().cpu().numpy()
            x_cpu = x_cpu * (moving_max - moving_min) + moving_min
            x_cpu_fits = fits.PrimaryHDU(x_cpu)
            gry = fits.HDUList(x_cpu_fits)
            gry.writeto(os.path.join(save_path, filename))

            end_time = time()  # 结束计时
            elapsed_time = end_time - start_time  # 配准时间
            total_time += elapsed_time  # 累加总时间
            num_pairs += 1  # 累计图像对数量

            print(f'Image pair {num_pairs}: Registration time = {elapsed_time:.2f} seconds')

            i += 1

        average_time = total_time / num_pairs  # 计算平均时间
        print(f'Average registration time per image pair = {average_time:.2f} seconds')
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))



if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()