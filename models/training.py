# import tensorflow as tf
import numpy as np
# from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from metrics.trends import calc_trend

class FastSimDataset(Dataset):
    def __init__(self, train=True, data = [], features = None, noise_power = None):
        super().__init__()
        self.train = train
        self.data = data
        self.features = features
        self.noise_power = noise_power

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.train:
          sample = self.data[item]
          feature = None
          if self.features is not None:
              feature = self.features[item]
              if self.noise_power is not None:
                  feature = (
                      feature
                      + np.random.normal(size=feature.shape).astype(feature.dtype) * self.noise_power
                  )
          return sample, feature
        else:
          sample = self.data[item]
          feature = None
          if self.features is not None:
              feature = self.features[item]
          return sample, feature


def calc_chi2(real, feature_real, gen, feature_gen):
    bins = np.linspace(min(feature_real.min(), feature_gen.min()), max(feature_real.max(), feature_gen.max()), 20)
    ((real_mean, real_std), (real_mean_err, real_std_err)) = calc_trend(
        feature_real, real, do_plot=False, bins=bins, window_size=1
    )
    ((gen_mean, gen_std), (gen_mean_err, gen_std_err)) = calc_trend(
        feature_gen, gen, do_plot=False, bins=bins, window_size=1
    )

    gen_upper = gen_mean + gen_std
    gen_lower = gen_mean - gen_std
    gen_err2 = gen_mean_err**2 + gen_std_err**2

    real_upper = real_mean + real_std
    real_lower = real_mean - real_std
    real_err2 = real_mean_err**2 + real_std_err**2

    chi2 = ((gen_upper - real_upper) ** 2 / (gen_err2 + real_err2)).sum() + (
        (gen_lower - real_lower) ** 2 / (gen_err2 + real_err2)
    ).sum()

    return chi2


def train(
    model,
    device,
    data_train,
    data_val,
    train_step_fn,
    loss_eval_fn,
    num_epochs,
    batch_size,
    train_writer=None,
    val_writer=None,
    callbacks=[],
    features_train=None,
    features_val=None,
    features_noise=None,
    first_epoch=0,
):
    if not ((features_train is None) or (features_val is None)):
        assert features_train is not None, 'train: features should be provided for both train and val'
        assert features_val is not None, 'train: features should be provided for both train and val'

    train_gen_losses = []
    train_disc_losses = []
    val_gen_losses = []
    val_disc_losses = []
    train_chi2 = []
    val_chi2 = []

    train_dataset = FastSimDataset(train=True, data=data_train, features=features_train, noise_power=features_noise)
    val_dataset = FastSimDataset(train=False, data=data_val, features=features_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    for i_epoch in range(first_epoch, num_epochs):
        print("Working on epoch #{}".format(i_epoch), flush=True)

        # model.train()
        
        shuffle_ids = np.random.permutation(len(data_train))
        losses_train = {}

        noise_power = None
        if features_noise is not None:
            noise_power = features_noise(i_epoch)

        for batch, feature_batch in tqdm(train_loader, desc='Train'):
            if features_train is None:
                losses_train_batch = train_step_fn(batch)
            else:
                # print('t', feature_batch.shape, batch.shape)
                losses_train_batch = train_step_fn(feature_batch, batch)
            for k, l in losses_train_batch.items():
                # print('l', l.item())
                losses_train[k] = losses_train.get(k, 0) + l.item() * len(batch)
        losses_train = {k: l / len(data_train) for k, l in losses_train.items()}
        train_gen_losses.append(losses_train['gen_loss'])
        train_disc_losses.append(losses_train['disc_loss'])

        # model.eval()

        losses_val = {}
        for batch, feature_batch in tqdm(val_loader, desc='Val'):
            if features_train is None:
                losses_val_batch = {k: l.cpu().detach().numpy() for k, l in loss_eval_fn(batch).items()}
            else:
                losses_val_batch = {k: l.cpu().detach().numpy() for k, l in loss_eval_fn(feature_batch, batch).items()}
            for k, l in losses_val_batch.items():
                losses_val[k] = losses_val.get(k, 0) + l * len(batch)
        losses_val = {k: l / len(data_val) for k, l in losses_val.items()}
        val_gen_losses.append(losses_val['gen_loss'])
        val_disc_losses.append(losses_val['disc_loss'])

        for f in callbacks:
            f(i_epoch)

        if train_writer is not None:
            with train_writer.as_default():
                for k, l in losses_train.items():
                    writer = SummaryWriter()
                    writer.add_scalar(k, l, i_epoch)
                    writer.close()

        if val_writer is not None:
            with val_writer.as_default():
                for k, l in losses_val.items():
                    writer = SummaryWriter()
                    writer.add_scalar(k, l, i_epoch)
                    writer.close()

        
        # fake_samples = model.make_fake(torch.tensor(features_val).to(device))
        # chi2 = calc_chi2(data_val, features_val, fake_samples, features_val)
        # val_chi2.append(chi2)
        fake_samples = model.make_fake(torch.tensor(features_train).to(device))
        # chi2 = calc_chi2(data_train, features_train, fake_samples, features_train)
        # train_chi2.append(chi2)

        print("", flush=True)
        print("Train losses:", losses_train)
        print("Val losses:", losses_val)
        # print("Train chi2:", train_chi2[-1])
        # print("Val chi2:", val_chi2[-1])

        if i_epoch % 50 == 0:
            np.savetxt('losses', np.array([train_gen_losses, train_disc_losses, val_gen_losses, val_disc_losses]))
            np.savetxt('chi2', np.array([train_chi2, val_chi2]))
            np.savetxt('train_samples', data_train.reshape((-1, 8 * 16)))
            np.savetxt('fake_samples', fake_samples.cpu().detach().numpy().reshape((-1, 8 * 16)))

    np.savetxt('losses', np.array([train_gen_losses, train_disc_losses, val_gen_losses, val_disc_losses]))
    np.savetxt('chi2', np.array([train_chi2, val_chi2]))
    np.savetxt('train_samples', data_train.reshape((-1, 8 * 16)))
    fake_samples = model.make_fake(torch.tensor(features_train).to(device))
    np.savetxt('fake_samples', fake_samples.cpu().detach().numpy().reshape((-1, 8 * 16)))


def average(models):
    parameters = [model.parameters() for model in models]
    assert len(np.unique([len(par) for par in parameters])) == 1, 'average: different models provided'

    result = torch.clone(models[0])
    for params in zip(result.parameters(), *parameters):
        params[0].assign(np.mean(params[1:], axis=0))

    return result
