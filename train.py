import torch
import losses
import os
import sys
from tqdm import tqdm
import numpy as np
from torch import optim
import torch.nn as nn
from models.Unet import DH_net
from torchvision import transforms
from models.STN import register_model
import matplotlib.pyplot as plt
from data import dataset, trans
from torch.utils.data import DataLoader
from skimage.metrics import  structural_similarity
from skimage.metrics import structural_similarity as ssim_skimage
from torch.utils.tensorboard import SummaryWriter
import natsort as ns
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

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

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1] = 1  # 在列方向上设置网格线
    for i in range(0, grid_img.shape[0], grid_step):
        grid_img[i+line_thickness-1, :] = 1  # 在行方向上设置网格线
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).float()  # 转换为 float 类型
    return grid_img

# def SSIM(img1, img2):
#
#     ssim = structural_similarity(img1, img2, data_range=img1.max() - img1.min())
#     return ssim
#
def comput_fig(img):
    img = img.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 12), dpi=180)
    num_images = img.shape[0]
    cols = 4
    rows = (num_images + cols - 1) // cols
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(img[i, 0, :, :], cmap='gray')  # 显示单通道灰度图像
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def SSIM(img1, img2):
    img1 = img1.squeeze().cpu().numpy()  # 将张量转换为 NumPy 数组
    img2 = img2.squeeze().cpu().numpy()  # 将张量转换为 NumPy 数组
    ssim_value = ssim_skimage(img1, img2, data_range=img1.max() - img1.min())
    return torch.tensor(ssim_value, dtype=torch.float32)


def main():
    device = torch.device(GPU_iden if torch.cuda.is_available() else "cpu")

    train_dir = 'F:/AI/TransMorph/dataset/data/'
    test_dir = 'F:/AI/TransMorph/dataset/data/test/'
    save_dir = 'cesiNewNET'
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)

    batch_size = 1
    weights = [1, 0.2]
    max_epoch = 12
    lr = 0.00001
    image_size = (512, 512)
    cont_training = False
    # ssim = SSIM(data_range=255, size_average=True, channel=1)
    model = DH_net(image_size)
    model.to(device)

    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        # Load the latest model
        model_files = ns.natsorted(os.listdir(model_dir))
        latest_model_file = model_files[-1]
        best_model_path = os.path.join(model_dir, latest_model_file)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model: {latest_model_file} loaded!")
    else:
        updated_lr = lr

    optimizer = optim.Adam(model.parameters(), lr= updated_lr, weight_decay= 0, amsgrad= True)


    criterion = nn.MSELoss()

    criterions = [criterion]
    criterions += [losses.Grad(penalty='l2')]
    reg_model = register_model(image_size, 'bilinear')

    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    train_set = dataset.ReadDataset(train_dir, train_composed)
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle=False, num_workers= 0, pin_memory= True)

    test_set = dataset.testReadDataset(test_dir, transforms=train_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir='logs/' + save_dir)
    for epoch in range(1, max_epoch):

        model.train()
        loss_all = 0.0

        for batch_idx, (moving, fixed) in enumerate(train_loader):

            moving, fixed = moving.to(device), fixed.to(device)
            input = torch.cat((moving, fixed), dim= 1)

            # optimizer.zero_grad()
            output = model(input)

            loss =0
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], fixed) * weights[n]
                loss += curr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss_all += loss.item()
        writer.add_scalar('Loss/train', loss_all / len(train_loader), epoch)
        print(f"Epoch {epoch}, Loss: {loss_all / len(train_loader)}")

        eval_det = AverageMeter()
        eval_ssim = AverageMeter()

        with torch.no_grad():

            for step, data in enumerate(test_loader):
                model.eval()

                x = data[0].to(device)
                y = data[1].to(device)
                moving_max = data[4].cpu().numpy()
                moving_min = data[5].cpu().numpy()
                x_in = torch.cat((x, y), dim=1)

                x_def, flow = model(x_in)
                jac_det = jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :])
                eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())

                grid_img = mk_grid_img(20, 1, (512, 512))
                grid_img = grid_img.to(device)
                flow = flow.to(device)
                def_flow = reg_model([grid_img.float(), flow])

                ncc = SSIM(x_def, y)
                eval_ssim.update(ncc.item(), y.numel())
                print(eval_ssim.avg)

        if epoch % 5 == 0:
            save_path = f"experiments/{save_dir}/model_epoch_{epoch}.pth"
            torch.save(
                {
                    'state_dict':model.state_dict(),
                },save_path)
            print(f"Saved model weights at epoch {epoch} to {save_path}")
        writer.add_scalar('DSC/validate', eval_ssim.avg, epoch)
        plt.switch_backend('agg')

        pred_fig = comput_fig(x_def)
        grid_fig = comput_fig(def_flow)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)

        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        # loss_all.reset()
    writer.close()

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