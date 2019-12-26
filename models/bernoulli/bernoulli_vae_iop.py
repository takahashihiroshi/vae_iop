import math
import time

import torch
from torch import nn

from models.utils import bernoulli_nll, gaussian_nll, standard_gaussian_nll, gaussian_kl_divergence, reparameterize
from models.utils import from_numpy, to_numpy, make_data_loader


class BernoulliNetwork(nn.Module):
    def __init__(self, n_in, n_latent, n_h):
        super(BernoulliNetwork, self).__init__()

        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h

        # Encoder
        self.le1 = nn.Sequential(
            nn.Linear(n_in, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        self.le2_mu = nn.Linear(n_h, n_latent)
        self.le2_ln_var = nn.Linear(n_h, n_latent)

        # Decoder
        self.ld1 = nn.Sequential(
            nn.Linear(n_latent, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        self.ld2_logits = nn.Linear(n_h, n_in)

    def encode(self, x):
        h = self.le1(x)
        return self.le2_mu(h), self.le2_ln_var(h)

    def decode(self, z):
        h = self.ld1(z)
        return self.ld2_logits(h)

    def forward(self, x):
        mu, ln_var = self.encode(x)
        return reparameterize(mu=mu, ln_var=ln_var)


class Discriminator(nn.Module):
    def __init__(self, n_latent, n_h):
        super(Discriminator, self).__init__()

        self.n_latent = n_latent
        self.n_h = n_h

        # Layer
        self.layers = nn.Sequential(
            nn.Linear(n_latent, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(), nn.Dropout(),
            nn.Linear(n_h, 1)
        )

    def forward(self, z):
        return self.layers(z).squeeze()


class BernoulliVAEIOP:

    def __init__(self, n_in, n_latent, n_h):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = BernoulliNetwork(n_in=n_in, n_latent=n_latent, n_h=n_h).to(self.device)
        self.discriminator = Discriminator(n_latent=n_latent, n_h=n_h).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_losses = []
        self.train_times = []
        self.reconstruction_errors = []
        self.kl_divergences = []
        self.valid_losses = []
        self.min_valid_loss = float("inf")

    def _compute_RE_and_KL(self, x, k=1):
        mu_enc, ln_var_enc = self.network.encode(x)

        RE = 0
        density_ratio = 0
        for i in range(k):
            z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            logits = self.network.decode(z)
            RE += bernoulli_nll(x, logits=logits) / k
            density_ratio += self.discriminator(z) / k

        KL = gaussian_kl_divergence(mu=mu_enc, ln_var=ln_var_enc) - density_ratio
        return RE, KL

    def _evidence_lower_bound(self, x, k=1):
        RE, KL = self._compute_RE_and_KL(x, k=k)
        return -1 * (RE + KL)

    def _importance_sampling(self, x, k=1):
        mu_enc, ln_var_enc = self.network.encode(x)
        lls = []
        for i in range(k):
            z = reparameterize(mu=mu_enc, ln_var=ln_var_enc)
            logits = self.network.decode(z)
            ll = -1 * bernoulli_nll(x, logits=logits, dim=1)
            ll -= standard_gaussian_nll(z, dim=1)
            ll += gaussian_nll(z, mu=mu_enc, ln_var=ln_var_enc, dim=1)
            ll += self.discriminator(z)
            lls.append(ll[:, None])

        return torch.cat(lls, dim=1).logsumexp(dim=1) - math.log(k)

    def _loss_VAE(self, x, k=1, beta=1):
        RE, KL = self._compute_RE_and_KL(x, k=k)
        RE_sum = RE.sum()
        KL_sum = KL.sum()
        loss = RE_sum + beta * KL_sum
        return loss, RE_sum, KL_sum

    def _loss_DRE(self, x):
        z_inferred = self.network(x).detach()
        z_sampled = torch.randn_like(z_inferred)
        logits_inferred = self.discriminator(z_inferred, use_dropout=True)
        logits_sampled = self.discriminator(z_sampled, use_dropout=True)
        loss = self.criterion(logits_inferred, torch.ones_like(logits_inferred))
        loss += self.criterion(logits_sampled, torch.zeros_like(logits_sampled))
        return loss

    def fit(self, X, k=1, batch_size=100,
            n_epoch_primal=500, n_epoch_dual=10,
            learning_rate_primal=1e-4, learning_rate_dual=1e-3,
            dynamic_binarization=False, warm_up=False, warm_up_epoch=100,
            is_stoppable=False, X_valid=None, path=None):

        self.network.train()
        self.discriminator.train()
        N = X.shape[0]
        data_loader = make_data_loader(X, device=self.device, batch_size=batch_size)
        optimizer_primal = torch.optim.Adam(self.network.parameters(), lr=learning_rate_primal)
        optimizer_dual = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_dual)

        if is_stoppable:
            X_valid = from_numpy(X_valid, self.device)

        for epoch_primal in range(n_epoch_primal):
            start = time.time()

            # warm-up
            beta = 1 * epoch_primal / warm_up_epoch if warm_up and epoch_primal <= warm_up_epoch else 1

            mean_loss = 0
            mean_RE = 0
            mean_KL = 0

            # Training VAE
            self.discriminator.eval()
            for _, batch in enumerate(data_loader):
                self.network.zero_grad()
                xs = torch.bernoulli(batch[0]) if dynamic_binarization else batch[0]
                loss_VAE, RE, KL = self._loss_VAE(xs, k=k, beta=beta)
                loss_VAE.backward()
                mean_loss += loss_VAE.item() / N
                mean_RE += RE.item() / N
                mean_KL += KL.item() / N
                optimizer_primal.step()

            # Training DRE
            self.discriminator.train()
            for epoch_dual in range(n_epoch_dual):
                sum_loss_DRE = 0
                for _, batch in enumerate(data_loader):
                    self.discriminator.zero_grad()
                    xs = torch.bernoulli(batch[0]) if dynamic_binarization else batch[0]
                    mu_enc, ln_var_enc = self.network.encode(x=xs)
                    z_inferred = reparameterize(mu_enc, ln_var_enc).detach()
                    z_sampled = torch.randn_like(z_inferred)
                    logits_inferred = self.discriminator(z_inferred)
                    logits_sampled = self.discriminator(z_sampled)
                    loss_DRE = self.criterion(logits_inferred, torch.ones_like(logits_inferred))
                    loss_DRE += self.criterion(logits_sampled, torch.zeros_like(logits_sampled))
                    loss_DRE.backward()
                    sum_loss_DRE += loss_DRE.item()
                    optimizer_dual.step()

                print(f"\tDRE epoch: {epoch_dual} / Train: {sum_loss_DRE / N:f}")

            end = time.time()
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.reconstruction_errors.append(mean_RE)
            self.kl_divergences.append(mean_KL)

            print(f"VAE epoch: {epoch_primal} / Train: {mean_loss:0.3f} / RE: {mean_RE:0.3f} / KL: {mean_KL:0.3f}", end='')

            if warm_up and epoch_primal < warm_up_epoch:
                print(" / Warm-up", end='')
            elif is_stoppable:
                valid_loss, _, _ = self._loss_VAE(X_valid, k=k, beta=1)
                valid_loss = valid_loss.item() / X_valid.shape[0]
                print(f" / Valid: {valid_loss:0.3f}", end='')
                self.valid_losses.append(valid_loss)
                self._early_stopping(valid_loss, path)

            print('')

        if is_stoppable:
            self.network.load_state_dict(torch.load(path))

        self.network.eval()
        self.discriminator.eval()

    def _early_stopping(self, valid_loss, path):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            torch.save(self.network.state_dict(), path)
            print(" / Save", end='')

    def encode(self, X):
        mu, ln_var = self.network.encode(from_numpy(X, self.device))
        return to_numpy(mu, self.device), to_numpy(ln_var, self.device)

    def decode(self, Z):
        logits = self.network.decode(from_numpy(Z, self.device))
        return to_numpy(logits, self.device)

    def evidence_lower_bound(self, X, k=1):
        return to_numpy(self._evidence_lower_bound(from_numpy(X, self.device), k=k), self.device)

    def importance_sampling(self, X, k=1):
        return to_numpy(self._importance_sampling(from_numpy(X, self.device), k=k), self.device)
