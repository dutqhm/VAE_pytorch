import torch
import torch.nn.functional as F
from .model import Encoder, Decoder
from .loss import VAE_loss, CVAE_su_loss

class VAE(torch.nn.Module):
    def __init__(self, filters, kernel_size, latent_dim, image_size, num_classes=10):
        super(VAE, self).__init__()
        self.encoder = Encoder(filters, kernel_size, latent_dim, image_size)
        self.decoder = Decoder(filters, kernel_size, latent_dim, image_size)
        self.loss    = VAE_loss()
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        loss = self.loss(x, x_recon, mu, logvar)
        return x_recon, mu, logvar, loss
    

class CVAE(VAE):
    def __init__(self, filters, kernel_size, latent_dim, image_size, num_classes=10):
        super(CVAE, self).__init__(filters, kernel_size, latent_dim, image_size, num_classes=num_classes)
        # self.encoder = Encoder(filters, kernel_size, latent_dim, image_size)
        self.num_classes = num_classes
        self.decoder = Decoder(filters, kernel_size, latent_dim + num_classes, image_size)
        
    def forward(self, x, y):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        y = F.one_hot(y, num_classes=self.num_classes).float()
        z = torch.cat((z, y), dim=1)
        
        x_recon = self.decoder(z)
        loss = self.loss(x, x_recon, mu, logvar)
        return x_recon, mu, logvar, loss
    

class CVAE_su(VAE):
    def __init__(self, filters, kernel_size, latent_dim, image_size, num_classes=10):
        super(CVAE_su, self).__init__(filters, kernel_size, latent_dim, image_size, num_classes=num_classes)

        self.num_classes = num_classes
        self.encoder_class = torch.nn.Linear(in_features=num_classes, out_features=latent_dim) ## 为每个类编码一个均值
        self.loss    = CVAE_su_loss()
        
    def forward(self, x, y):
        mu, logvar = self.encoder(x)
        
        y = F.one_hot(y, num_classes=self.num_classes).float()
        yh = self.encoder_class(y)
        
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        loss = self.loss(x, x_recon, mu, logvar, yh)
        return x_recon, mu, logvar, loss
    



