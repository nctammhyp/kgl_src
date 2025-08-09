import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.grad_factor = 10.
        self.normal_factor = 1.

       
    def forward(self,criterion,pred,target,epoch=0):
        if 'l1' in criterion:
            depth_loss = self.L1_imp_Loss(pred,target)
        elif 'l2' in criterion:
            depth_loss = self.L2_imp_Loss(pred,target)
        elif 'rmsle' in criterion:
            depth_loss = self.RMSLELoss(pred,target)
        if 'gn' in criterion:
            grad_target, grad_pred = self.imgrad_yx(target), self.imgrad_yx(pred)
            grad_loss = self.GradLoss(grad_pred, grad_target) * self.grad_factor
            normal_loss = self.NormLoss(grad_pred, grad_target) * self.normal_factor
            return depth_loss + grad_loss + normal_loss
        else:
            return depth_loss
    
    def GradLoss(self,grad_target,grad_pred):
        return torch.sum( torch.mean( torch.abs(grad_target-grad_pred) ) )
    
    def NormLoss(self, grad_target, grad_pred):
        prod = ( grad_pred[:,:,None,:] @ grad_target[:,:,:,None] ).squeeze(-1).squeeze(-1)
        pred_norm = torch.sqrt( torch.sum( grad_pred**2, dim=-1 ) )
        target_norm = torch.sqrt( torch.sum( grad_target**2, dim=-1 ) ) 
        return 1 - torch.mean( prod/(pred_norm*target_norm) )
    
    def RMSLELoss(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 0.5), torch.log(target + 0.5)))
 
        
    
    def L1_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
    def L2_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss
    
    def imgrad_yx(self,img):
        N,C,_,_ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)
    
    def imgrad(self,img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x


class ScaleInvariantGradientMatchingLoss(nn.Module):
    def __init__(self, scales=(1,2,4), loss_type='l1', eps=1e-6):
        super().__init__()
        self.scales = scales
        self.loss_type = loss_type
        self.eps = eps

        kernel_x = torch.tensor([[[[-1, 1]]]], dtype=torch.float32)
        kernel_y = torch.tensor([[[[-1], [1]]]], dtype=torch.float32)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        h = x.shape[-2] // factor
        w = x.shape[-1] // factor
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def _gradient_xy(self, x):
        pad_x = (1, 0, 0, 0)
        pad_y = (0, 0, 1, 0)
        gx = F.conv2d(F.pad(x, pad_x, mode='replicate'),
                      self.kernel_x.repeat(x.shape[1], 1, 1, 1),
                      groups=x.shape[1])
        gy = F.conv2d(F.pad(x, pad_y, mode='replicate'),
                      self.kernel_y.repeat(x.shape[1], 1, 1, 1),
                      groups=x.shape[1])
        return gx, gy

    def forward(self, pred, target, valid_mask):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if valid_mask is None:
            valid_mask = torch.ones_like(pred, dtype=torch.bool, device=pred.device)
        else:
            if valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(torch.bool)

        pred = pred.clamp(min=self.eps)
        target = target.clamp(min=self.eps)

        log_pred_full = torch.log(pred)
        log_target_full = torch.log(target)

        total_loss = 0.0
        for factor in self.scales:
            lp = self._downsample(log_pred_full, factor)
            lt = self._downsample(log_target_full, factor)
            vm = self._downsample(valid_mask.float(), factor) >= 0.5

            gx_p, gy_p = self._gradient_xy(lp)
            gx_t, gy_t = self._gradient_xy(lt)

            diff_x = gx_p - gx_t
            diff_y = gy_p - gy_t

            if self.loss_type == 'l1':
                per_px = torch.abs(diff_x) + torch.abs(diff_y)
            else:
                per_px = diff_x ** 2 + diff_y ** 2

            mask = vm.expand_as(per_px)
            valid_count = mask.float().sum()
            if valid_count.item() > 0:
                loss_scale = (per_px * mask.float()).sum() / (valid_count + self.eps)
                total_loss += loss_scale

        return total_loss / len(self.scales)


class CombinedDepthLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, lambd_silog=0.5):
        super().__init__()
        self.silog = SiLogLoss(lambd=lambd_silog)
        self.grad_loss = ScaleInvariantGradientMatchingLoss(scales=(1, 2, 4))
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target, valid_mask):
        loss_silog = self.silog(pred, target, valid_mask)
        loss_grad = self.grad_loss(pred, target, valid_mask)
        total = self.alpha * loss_silog + self.beta * loss_grad
        return total, loss_silog, loss_grad
