from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.indices = args

    def forward(self, x):
        return x.permute(self.indices)


class VariationalSpatialTransformer(nn.Module):
    def __init__(self, loc_layer, image_shape, target_shape, num_heads=32, d_loc=128, mask_layer=None, initial_sigma=1.0):
        super(VariationalSpatialTransformer, self).__init__()
        
        self.num_heads=num_heads
        self.target_shape = target_shape
        
        if mask_layer is None:
            mask_layer = torch.nn.Conv2d(in_channels=image_shape[0], out_channels=32, kernel_size=3, padding="same")
        with torch.no_grad():
            mask_layer_channels = mask_layer(torch.rand((1, image_shape[0], image_shape[1], image_shape[2]))).view(1, -1, image_shape[1], image_shape[2]).shape[1]
        self.mask_layer = nn.Sequential(
            mask_layer,
            nn.ReLU(),
            Permute(0, 2, 3, 1),
            nn.Linear(
                mask_layer_channels,
                image_shape[0]
            ),
            nn.Sigmoid(),
            Permute(0, 3, 1, 2),
        )
        
        # Spatial transformer localization-network
        self.loc_layer = loc_layer

        # Regressor for the 3 * 2 affine matrix
        with torch.no_grad():
            self.fc_loc_input_features = self.loc_layer(torch.rand((1, image_shape[0], image_shape[1], image_shape[2]))).view(1, -1).shape[1]
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_input_features, d_loc),
            nn.ReLU(),
            nn.Linear(d_loc, 2 * self.num_heads * 2 * 3 * 2)  # ([mu, sigma], num_heads, [affine_1, affine_2], 3, 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        with torch.no_grad():
            initial_mu_x = (target_shape[2] / image_shape[2])
            initial_mu_y = (target_shape[1] / image_shape[1])
            
            initial_sigma_x = torch.log(torch.tensor(initial_mu_x)).item()
            initial_sigma_y = torch.log(torch.tensor(initial_mu_y)).item()
            initial_sigma_aff = torch.log(torch.tensor(initial_sigma)).item()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor(
                [
                    1, 0, 0,
                    0, 1, 0,
                    initial_mu_y, 0, 0,
                    0, initial_mu_x, 0
                ]*self.num_heads
                + [
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_aff,
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_aff,
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_y,
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_x
                ]*self.num_heads,
                dtype=torch.float
            )
        )
        
    def sample_z(self, batch_size):
        return torch.normal(
            torch.zeros((batch_size, 2*self.num_heads*2*3)),
            torch.ones((batch_size, 2*self.num_heads*2*3)),
        ).view(batch_size, self.num_heads, 2, 2, 3)
        
    def reparametrize(self, z, mu, log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        return mu + sigma*z.to(sigma.device)

    # Spatial transformer network forward function
    def forward(self, x, z=None):
        xs = self.loc_layer(x)
        xs = xs.view(-1, self.fc_loc_input_features)
        theta_mu_sigma = self.fc_loc(xs)
        theta_mu_sigma = theta_mu_sigma.view(-1, 2, self.num_heads, 2, 2, 3)
        if z is None:
            z = self.sample_z(x.shape[0])
        theta_mu, theta_sigma = theta_mu_sigma[:, 0, :, :, :, :], theta_mu_sigma[:, 1, :, :, :, :]
        samples = []
        for i in range(self.num_heads):
            theta_0 = self.reparametrize(z[:, i, 0], theta_mu[:, i, 0], theta_sigma[:, i, 0])
            grid = F.affine_grid(theta_0, x.size(), align_corners=False)
            x_crop = F.grid_sample(x, grid, align_corners=False)
            x_crop = x_crop * self.mask_layer(x_crop)
            theta_1 = self.reparametrize(z[:, i, 1], theta_mu[:, i, 1], theta_sigma[:, i, 1])
            grid = F.affine_grid(theta_1, (x.shape[0],) + self.target_shape, align_corners=False)
            samples.append(F.grid_sample(x_crop, grid, align_corners=False))
        x = reduce(torch.add, samples)
        return x