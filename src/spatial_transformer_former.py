from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalSpatialTransformer(nn.Module):
    def __init__(self, loc_layer, image_shape, target_shape, num_samples=32, d_loc=128, initial_sigma=1.0):
        super(VariationalSpatialTransformer, self).__init__()
        
        self.num_samples=num_samples
        self.target_shape = target_shape
        
        # Spatial transformer localization-network
        self.loc_layer = loc_layer

        # Regressor for the 3 * 2 affine matrix
        with torch.no_grad():
            self.fc_loc_input_features = self.loc_layer(torch.rand((1, image_shape[0], image_shape[1], image_shape[2]))).view(1, -1).shape[1]
        print(self.fc_loc_input_features)
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_input_features, d_loc),
            nn.ReLU(),
            nn.Linear(d_loc, 2 * self.num_samples * 2 * 3 * 2)  # ([mu, sigma], num_samples, [affine_1, affine_2], 3, 2)
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
                ]*self.num_samples
                + [
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_aff,
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_aff,
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_y,
                    initial_sigma_aff, initial_sigma_aff, initial_sigma_x
                ]*self.num_samples,
                dtype=torch.float
            )
        )
        
    def sample_z(self, batch_size):
        return torch.normal(
            torch.zeros((batch_size, 2*self.num_samples*2*3)),
            torch.ones((batch_size, 2*self.num_samples*2*3))
        ).view(batch_size, self.num_samples, 2, 2, 3)
        
    def reparametrize(self, z, mu, log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        return mu + sigma*z

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.loc_layer(x)
        xs = xs.view(-1, self.fc_loc_input_features)
        theta_mu_sigma = self.fc_loc(xs)
        theta_mu_sigma = theta_mu_sigma.view(-1, 2, self.num_samples, 2, 2, 3)
        z = self.sample_z(x.shape[0])
        theta_mu, theta_sigma = theta_mu_sigma[:, 0, :, :, :, :], theta_mu_sigma[:, 1, :, :, :, :]
        samples = []
        for i in range(self.num_samples):
            theta_0 = self.reparametrize(z[:, i, 0], theta_mu[:, i, 0], theta_sigma[:, i, 0])
            grid = F.affine_grid(theta_0, x.size(), align_corners=False)
            x_crop = F.grid_sample(x, grid, align_corners=False)
            
            theta_1 = self.reparametrize(z[:, i, 1], theta_mu[:, i, 1], theta_sigma[:, i, 1])
            grid = F.affine_grid(theta_1, (x.shape[0],) + self.target_shape, align_corners=False)
            samples.append(F.grid_sample(x_crop, grid, align_corners=False))
        x = reduce(torch.add, samples)
        return x