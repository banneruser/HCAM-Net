import torch
import numpy as np
import torch.nn.functional as F
import math

class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad



class MINDLoss(torch.nn.Module):
    def __init__(self, patch_size, neigh_size, sigma, eps):
        super(MINDLoss, self).__init__()
        self.patch_size = patch_size
        self.neigh_size = neigh_size
        self.sigma = sigma
        self.eps = eps

    def gaussian_kernel(self, sigma, sz):
        xpos_vec = np.arange(sz)
        ypos_vec = np.arange(sz)
        output = np.ones([sz, sz], dtype=np.single)
        midpos = sz // 2
        for xpos in xpos_vec:
            for ypos in ypos_vec:
                output[xpos, ypos] = np.exp(-((xpos - midpos) ** 2 + (ypos - midpos) ** 2) / (2 * sigma ** 2)) / (
                            2 * np.pi * sigma ** 2)
        return output

    def Dp(self, image, xshift, yshift):
        shift_image = self.torch_image_translate(image, xshift, yshift, interpolation='nearest')
        diff = torch.sub(image, shift_image)
        diff_square = torch.mul(diff, diff)
        res = F.conv2d(diff_square,
                       weight=torch.from_numpy(self.gaussian_kernel(self.sigma, self.patch_size)).unsqueeze(0).unsqueeze(0), stride=1,
                       padding=3)
        return res.squeeze(0)

    def torch_image_translate(self, input_, tx, ty, interpolation='nearest'):
        input_shape = input_.size()
        if min(input_shape[2:]) <= 1:
            raise ValueError("输入图像的大小不能小于等于1")

        # 构建平移矩阵
        translation_matrix = torch.eye(3, dtype=torch.float)  # 3x3 矩阵
        translation_matrix[0, 2] = tx  # 设置 x 方向的平移量
        translation_matrix[1, 2] = ty  # 设置 y 方向的平移量

        # 创建二维仿射变换矩阵，适应 affine_grid 的要求
        affine_matrix = translation_matrix[:2].unsqueeze(0)  # 取前两行，并添加一个批次维度

        # 生成网格
        grid = F.affine_grid(affine_matrix, input_shape, align_corners=False)
        # 对输入图像进行采样
        wrp = F.grid_sample(input_, grid, mode=interpolation, padding_mode='zeros', align_corners=False)

        return wrp

    def forward(self, image1, image2):
        reduce_size = int((self.patch_size + self.neigh_size - 2) / 2)

        Vimg = torch.add(self.Dp(image1, -1, 0), self.Dp(image1, 1, 0))
        Vimg = torch.add(Vimg, self.Dp(image1, 0, -1))
        Vimg = torch.add(Vimg, self.Dp(image1, 0, 1))
        Vimg = torch.div(Vimg, 4) + torch.mul(torch.ones_like(Vimg), self.eps)

        xshift_vec = np.arange(-(self.neigh_size // 2), self.neigh_size - (self.neigh_size // 2))
        yshift_vec = np.arange(-(self.neigh_size // 2), self.neigh_size - (self.neigh_size // 2))

        iter_pos = 0
        for xshift in xshift_vec:
            for yshift in yshift_vec:
                if (xshift, yshift) == (0, 0):
                    continue
                MIND_tmp = torch.exp(torch.mul(torch.div(self.Dp(image2, xshift, yshift), Vimg), -1))
                tmp = MIND_tmp[:, reduce_size:(image1.size()[1] - reduce_size), reduce_size:(image1.size()[2] - reduce_size)]
                if iter_pos == 0:
                    output = tmp
                else:
                    output = torch.cat([output, tmp], 0)
                iter_pos = iter_pos + 1

        input_max, input_indexes = torch.max(output, dim=0)
        output = torch.div(output, input_max.unsqueeze(0))

        return torch.mean(output)



class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

# 使用示例
# mind_loss = MINDLoss(patch_size=7, neigh_size=3, sigma=1.0, eps=1e-6)
# loss = mind_loss(image1, image2)

# class MIND(torch.nn.Module):
#
#
#     def