import torch
import torch.nn.functional as F


class VAE_loss(torch.nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()
    
    def forward(self, x, x_recon, mu, logvar):
        batch_size = x.size(0)
        ## BCE loss
        # recon_loss = torch.nn.functional.binary_cross_entropy(x_recon, x,reduction='sum', size_average=False)
        ## MSE loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ## batch_size, 2
        return recon_loss/ batch_size, kl_div / batch_size
    
class CVAE_su_loss(torch.nn.Module):
    def __init__(self):
        super(CVAE_su_loss, self).__init__()
    
    def forward(self, x, x_recon, mu, logvar, yh):
        batch_size = x.size(0)
        ## BCE loss
        # recon_loss = torch.nn.functional.binary_cross_entropy(x_recon, x,reduction='sum', size_average=False)
        ## MSE loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - (mu - yh).pow(2) - logvar.exp()) ## batch_size, 2
        return recon_loss/ batch_size, kl_div / batch_size