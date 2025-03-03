import h5py
# import tensorflow as tf
# from tensorflow.python.keras.saving import hdf5_format
import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd
from torchsummary import summary


from . import scalers, nn


def preprocess_features(features):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    bin_fractions = features[:, 2:4] % 1
    features = (features[:, :3] - torch.tensor([[0.0, 0.0, 162.5]])) / torch.tensor([[20.0, 60.0, 127.5]])
    return torch.cat((features, bin_fractions), dim=-1)


def preprocess_features_v4plus(features, device):
    # features:
    #   crossing_angle [-20, 20]
    #   dip_angle [-60, 60]
    #   drift_length [35, 290]
    #   pad_coordinate [40-something, 40-something]
    #   padrow {23, 33}
    #   pT [0, 2.5]

    # print(features)
    bin_fractions = features[:, 2:4] % 1
    features_1 = (features[:, :3] - torch.tensor([[0.0, 0.0, 162.5]]).to(device)) / torch.tensor([[20.0, 60.0, 127.5]]).to(device)
    features_2 = (features[:, 4:5] >= 27)
    features_3 = features[:, 5:6] / 2.5
    return torch.cat((features_1, features_2, features_3, bin_fractions), dim=-1)


def disc_loss(d_real, d_fake):
    return torch.mean(d_fake - d_real)


def gen_loss(d_real, d_fake):
    return torch.mean(d_real - d_fake)


def disc_loss_cramer(d_real, d_fake, d_fake_2):
    return -torch.mean(
        torch.norm(d_real - d_fake, dim=-1)
        + torch.norm(d_fake_2, dim=-1)
        - torch.norm(d_fake - d_fake_2, dim=-1)
        - torch.norm(d_real, dim=-1)
    )


def gen_loss_cramer(d_real, d_fake, d_fake_2):
    return -disc_loss_cramer(d_real, d_fake, d_fake_2)


def logloss(x):
    sp = torch.nn.Softplus()
    return sp(-x)


def disc_loss_js(d_real, d_fake):
    # print('j', d_real.shape, d_fake.shape)
    return torch.sum(logloss(d_real)) + torch.sum(logloss(-d_fake)) / (len(d_real) + len(d_fake))


def gen_loss_js(d_real, d_fake):
    return torch.mean(logloss(d_fake))


class Model_v4:
    def __init__(self, device, config):
        self.device = device
        print("Device:", device)
        self._f = preprocess_features
        if config['data_version'] == 'data_v4plus':
            self.full_feature_space = config.get('full_feature_space', False)
            self.include_pT_for_evaluation = config.get('include_pT_for_evaluation', False)
            if self.full_feature_space:
                self._f = preprocess_features_v4plus

        self.gp_lambda = config['gp_lambda']
        self.gpdata_lambda = config['gpdata_lambda']
        self.num_disc_updates = config['num_disc_updates']
        self.cramer = config['cramer']
        self.js = config.get('js', False)
        assert not (self.js and self.cramer)

        self.stochastic_stepping = config['stochastic_stepping']
        self.dynamic_stepping = config.get('dynamic_stepping', False)
        if self.dynamic_stepping:
            assert not self.stochastic_stepping
            self.dynamic_stepping_threshold = config['dynamic_stepping_threshold']

        self.latent_dim = config['latent_dim']

        architecture_descr = config['architecture']

        self.generator = nn.FullModel(
            architecture_descr['generator'], custom_objects_code=config.get('custom_objects', None)
        ).to(device)
        self.discriminator = nn.FullModel(
            architecture_descr['discriminator'], custom_objects_code=config.get('custom_objects', None)
        ).to(device)

        self.disc_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=config['lr_disc'])
        self.gen_opt = torch.optim.RMSprop(self.generator.parameters(), lr=config['lr_gen'])

        self.step_counter = torch.tensor(0, dtype=torch.int)

        self.scaler = scalers.get_scaler(config['scaler'])
        self.pad_range = tuple(config['pad_range'])
        self.time_range = tuple(config['time_range'])
        self.data_version = config['data_version']

        # self.generator.compile(optimizer=self.gen_opt, loss='mean_squared_error')
        # self.discriminator.compile(optimizer=self.disc_opt, loss='mean_squared_error')

    def load_generator(self, checkpoint):
        self._load_weights(checkpoint, 'gen')

    def load_discriminator(self, checkpoint):
        self._load_weights(checkpoint, 'disc')

    def _load_weights(self, checkpoint, gen_or_disc):
        if gen_or_disc == 'gen':
            network = self.generator
            step_fn = self.gen_step
        elif gen_or_disc == 'disc':
            network = self.discriminator
            step_fn = self.disc_step
        else:
            raise ValueError(gen_or_disc)

        model_file = h5py.File(checkpoint, 'r')
        if len(network.optimizer.weights) == 0 and 'optimizer_weights' in model_file:
            # perform single optimization step to init optimizer weights
            features_shape = self.discriminator.inputs[0].shape.as_list()
            targets_shape = self.discriminator.inputs[1].shape.as_list()
            features_shape[0], targets_shape[0] = 1, 1
            step_fn(torch.ones(features_shape), torch.ones(targets_shape))
            # step_fn(torch.zeros(features_shape), torch.zeros(targets_shape))

        print(f'Loading {gen_or_disc} weights from {str(checkpoint)}')
        network.load_weights(str(checkpoint))

        if 'optimizer_weights' in model_file:
            print('Also recovering the optimizer state')
            opt_weight_values = hdf5_format.load_optimizer_weights_from_hdf5_group(model_file) #????
            network.optimizer.set_weights(opt_weight_values)

    def make_fake(self, features):
        # print('i', type(features))
        size = len(features)
        latent_input = torch.normal(mean=0, std=1, size=(size, self.latent_dim)).to(self.device)
        # print(features)
        # print(self._f(features))
        # print(torch.cat((self._f(features), latent_input), dim=-1))
        # print(self.generator(torch.cat((self._f(features), latent_input), dim=-1)))
        return self.generator(torch.cat((self._f(features, self.device), latent_input), dim=-1))

    def gradient_penalty(self, features, real, fake):
        alpha = torch.rand(size=[len(real)] + [1] * (len(real.shape) - 1)).to(self.device)
        # print('e', real.shape, fake.shape)
        fake = torch.reshape(fake, real.shape)
        interpolates = alpha * real + (1 - alpha) * fake
        
        inputs = [Variable(self._f(features, self.device), requires_grad=True), Variable(interpolates, requires_grad=True)]
        d_int = self.discriminator(inputs)
        # print(type(d_int), type(inputs[0]))
        # print('k', d_int.shape)
        # print('kk', d_int.grad)
        # print(d_int.shape, interpolates.shape)
        # grads = torch.gradient(d_int)
        grads = autograd.grad(outputs=d_int, inputs=inputs,
                               grad_outputs=torch.ones(
                                   d_int.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]
        return torch.mean(torch.maximum(torch.norm(grads, dim=-1) - 1, torch.Tensor([0]).to(self.device)) ** 2) 

    def gradient_penalty_on_data(self, features, real):
        d_real = self.discriminator([self._f(features), real])

        grads = torch.reshape(d_real.grad, (len(real), -1))
        return torch.mean(torch.sum(grads**2, dim=-1))


    def calculate_losses(self, feature_batch, target_batch):
        feature_batch = feature_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        fake = self.make_fake(feature_batch)
        d_real = self.discriminator([self._f(feature_batch, self.device), target_batch])
        d_fake = self.discriminator([self._f(feature_batch, self.device), fake])
        if self.cramer:
            fake_2 = self.make_fake(feature_batch)
            d_fake_2 = self.discriminator([self._f(feature_batch, self.device), fake_2])

        if not self.cramer:
            if self.js:
                d_loss = disc_loss_js(d_real, d_fake)
            else:
                d_loss = disc_loss(d_real, d_fake)
        else:
            d_loss = disc_loss_cramer(d_real, d_fake, d_fake_2)

        if self.gp_lambda > 0:
            d_loss = d_loss + self.gradient_penalty(feature_batch, target_batch, fake) * self.gp_lambda
        if self.gpdata_lambda > 0:
            d_loss = d_loss + self.gradient_penalty_on_data(feature_batch, target_batch) * self.gpdata_lambda
        if not self.cramer:
            if self.js:
                g_loss = gen_loss_js(d_real, d_fake)
            else:
                g_loss = gen_loss(d_real, d_fake)
        else:
            g_loss = gen_loss_cramer(d_real, d_fake, d_fake_2)

        return {'disc_loss': d_loss, 'gen_loss': g_loss}

    def disc_step(self, feature_batch, target_batch):
        feature_batch = torch.Tensor(feature_batch).to(self.device)
        target_batch = torch.Tensor(target_batch).to(self.device)

        for p in self.discriminator.parameters():
                p.requires_grad = True

        self.discriminator.zero_grad()
        # print('u', feature_batch.shape, target_batch.shape)
        losses = self.calculate_losses(feature_batch, target_batch)
        
        # self.disc_opt.zero_grad()
        losses['disc_loss'].backward()
        # losses['gen_loss'].backward()  
        self.disc_opt.step()
        return losses

    def gen_step(self, feature_batch, target_batch):
        feature_batch = torch.Tensor(feature_batch).to(self.device)
        target_batch = torch.Tensor(target_batch).to(self.device)

        for p in self.discriminator.parameters():
                p.requires_grad = False

        self.generator.zero_grad()
        losses = self.calculate_losses(feature_batch, target_batch)

        # self.gen_opt.zero_grad()
        # losses['disc_loss'].backward()
        losses['gen_loss'].backward()        
        self.gen_opt.step()
        return losses

    
    def training_step(self, feature_batch, target_batch):
        if self.stochastic_stepping:
            if torch.randint(high=self.num_disc_updates + 1, size=(1,))[0] == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
            else:
                result = self.disc_step(feature_batch, target_batch)
        else:
            if self.step_counter == self.num_disc_updates:
                result = self.gen_step(feature_batch, target_batch)
                self.step_counter.assign(0)
            else:
                result = self.disc_step(feature_batch, target_batch)
                if self.dynamic_stepping:
                    if result['disc_loss'] < self.dynamic_stepping_threshold:
                        self.step_counter.assign(self.num_disc_updates)
                else:
                    self.step_counter.assign_add(1)
        return result
