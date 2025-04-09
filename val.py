from astropy.io import fits
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from models.STN import register_model
from torchvision import transforms
from data import dataset, trans
from models.Unet import DH_net
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_edt
from skimage.metrics import  structural_similarity
# import torchmetrics

def SSIM(img1, img2):
    img1 = img1.squeeze().cpu().numpy()
    img2 =img2.squeeze().cpu().numpy()
    ssim = structural_similarity(img1, img2, data_range=img1.max() - img1.min())
    return ssim


def r_squared(y_true, y_pred):
    y_true = y_true.squeeze().cpu().numpy()
    y_pred = y_pred.squeeze().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

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

    img_size = (512, 512)
    model = DH_net()
    checkpoint = torch.load('experiments/1225newNet/model_epoch_195.pth')['state_dict']
    # state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    reg_model = register_model(img_size, 'bilinear')
    reg_model.cuda()

    test_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
    ])
    test_set = dataset.ReadDataset(test_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_det = AverageMeter()
    SSIM_det = AverageMeter()
    R2_det = AverageMeter()

    i = 1
    with torch.no_grad():

        for step, data in enumerate(tqdm(test_loader)):

            model.eval()

            x = data[0].to(device)
            y = data[1].to(device)

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)

            jac_det = jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())

            ssim_each = SSIM(x_def, y)
            SSIM_det.update(ssim_each)

            r2_each = r_squared(y, x_def)
            R2_det.update(r2_each)


            i += 1
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('SSIM det: {}, std: {}'.format(SSIM_det.avg, SSIM_det.std))
        print('R2 det: {}, std: {}'.format(R2_det.avg, R2_det.std))


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