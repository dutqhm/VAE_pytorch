import torch



class Encoder(torch.nn.Module):
    def __init__(self, filters, kernel_size, latent_dim, image_size):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=filters, out_channels=filters*2, kernel_size=kernel_size, stride=2, padding=1)
        self.fc = torch.nn.Linear(in_features=2*filters*(image_size//4)**2, out_features=filters)
        
        self.mean = torch.nn.Linear(in_features=filters, out_features=latent_dim)
        self.varlog = torch.nn.Linear(in_features=filters, out_features=latent_dim)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mean = self.mean(x)
        varlog = self.varlog(x)
        return mean, varlog
    
class Decoder(torch.nn.Module):
    def __init__(self, filters, kernel_size, latent_dim, image_size):
        super(Decoder, self).__init__()
        self.shape = (filters*2, image_size//4, image_size//4)
        self.fc = torch.nn.Linear(in_features=latent_dim, out_features=2*filters*(image_size//4)**2)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=filters*2, out_channels=filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=filters, out_channels=1, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.sigmoid(self.conv1(x))
        return x
    