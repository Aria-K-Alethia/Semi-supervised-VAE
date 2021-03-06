'''
    Copyright (c) 2020 Aria-K-Alethia@github.com

    Description:
        model definition
    Licence:
        MIT
    THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
    Any use of this code should display all the info above.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.distributions as dist
from utils import onehot_vector, unsupervised_label

REDUCTION = 'sum'
REDUCTION_METHOD = torch.sum

class Classifier(nn.Module):
    """
    """
    def __init__(self, x, y, c):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pipe = nn.Sequential(
                nn.Linear(x, c[0]),
                self.relu,
                nn.Linear(c[0], c[1]),
                self.relu,
                nn.Linear(c[1], y),
                nn.Softmax(dim=-1)
            )
    def forward(self, x):
        return self.pipe(x)

class VAE(nn.Module):
    def __init__(self, x, y, z, h):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(x, h)
        self.fc21 = nn.Linear(h, z)
        self.fc22 = nn.Linear(h, z)
        self.fc3 = nn.Linear(z, h)
        self.fc4 = nn.Linear(h, x)
        self.loc = nn.Parameter(torch.zeros(20), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(20), requires_grad=False)
        self.prior = dist.Independent(dist.Normal(self.loc, self.scale), 1)
        self.y = y
        self.x = x
        self.z = z
        self.h = h

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), F.softplus(self.fc22(h1))

    def reparameterize(self, mu, var):
        std = var
        pred_dist = dist.Independent(dist.Normal(mu, std), 1)
        self.pred_dist = pred_dist
        eps = pred_dist.rsample()
        return eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def reconstruct(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        return self.decode(z), {'mean': mu, 'var': var}

    def sample(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        return z

    def forward(self, x, label=None):
        x = x.view(-1, self.x)
        out, state = self.reconstruct(x)
        loss_state = {}
        loss_state['reconstruction'] = reconstruction_loss = F.binary_cross_entropy(out, x, reduction=REDUCTION)
        loss_state['kl'] = kl_loss = REDUCTION_METHOD(dist.kl_divergence(self.pred_dist, self.prior))
        loss_state['classification'] = loss_state['prior'] = torch.scalar_tensor(0)
        loss = reconstruction_loss + kl_loss
        return out, loss, loss_state['classification'], loss_state, state

class CVAE(nn.Module):
    def __init__(self, x, y, z, h, c):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(x + y, h)
        self.fc21 = nn.Linear(h, z)
        self.fc22 = nn.Linear(h, z)
        self.fc3 = nn.Linear(z + y, h)
        self.fc4 = nn.Linear(h, x)
        self.loc = nn.Parameter(torch.zeros(z), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(z), requires_grad=False)
        self.prior = dist.Independent(dist.Normal(self.loc, self.scale), 1)
        self.classifier = Classifier(x, y, c)
        self.y = y
        self.x = x
        self.z = z
        self.h = h
        self.c = c

    def classify(self, x):
        return self.classifier(x)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), F.softplus(self.fc22(h1))

    def reparameterize(self, mu, var):
        pred_dist = dist.Independent(dist.Normal(mu, var), 1)
        self.pred_dist = pred_dist
        eps = pred_dist.rsample()
        return eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def reconstruct(self, x, label):
        label_onehot = onehot_vector(label, self.y).type_as(x)
        mu, var = self.encode(torch.cat([x, label_onehot], dim=1))
        z = self.reparameterize(mu, var)
        z = torch.cat([z, label_onehot], dim=1)
        return self.decode(z), {'mean': mu, 'var': var}

    def forward(self, x, label=None, target=None):
        supervised = (label is not None)
        x = x.view(-1, self.x)
        if not supervised:
            label = unsupervised_label(x, self.y)
            xs = x.repeat(self.y, 1)
            if target is not None:
                target = target.repeat(self.y, 1)
        else:
            xs = x
        out, state = self.reconstruct(xs, label)
        reconstruction_loss = F.binary_cross_entropy(out, xs if target is None else target, reduction='none').sum(dim=-1)
        loss = reconstruction_loss
        # kl and prior
        kl_loss = dist.kl_divergence(self.pred_dist, self.prior)
        loss = loss + kl_loss
        prior_loss = torch.log2(torch.scalar_tensor(1/self.y)).to(kl_loss.device).type_as(kl_loss)
        loss = loss - prior_loss

        # classification
        if supervised:
            prob = self.classify(x) # [B, C]
            classification_loss = F.nll_loss(torch.log(prob), label, reduction=REDUCTION)
        else:
            classification_loss = 0
        loss_state = {'reconstruction': REDUCTION_METHOD(reconstruction_loss), 'kl': REDUCTION_METHOD(kl_loss), 'prior': prior_loss, 'classification': classification_loss}
        # if supervised, return current loss
        if supervised:
            return out, REDUCTION_METHOD(loss), classification_loss, loss_state, state
        # otherwise, compute the unsupervised loss
        prob = self.classify(x)
        loss = loss.view_as(prob.t()).t()

        # entropy
        H = torch.sum(prob * torch.log2(prob + 1e-8), dim=-1)
        L = torch.sum(prob * loss, dim=-1)

        loss = REDUCTION_METHOD(H + L)
        return out, loss, classification_loss, loss_state, state

class StackedVAE(CVAE):
    """
    """
    def __init__(self, x, y, z, h, c, vae):
        super(StackedVAE, self).__init__(vae.z, y, z, h, c)
        self.vae = vae
        in_feat = self.fc4.in_features
        self.fc4 = nn.Linear(in_feat, x)

        self.vae.train(False)
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x, label=None):
        latent = self.vae.sample(x)
        return super(StackedVAE, self).forward(latent, label, x)

class GMVAE(nn.Module):
    def __init__(self, x, y, z, h, c):
        super(GMVAE, self).__init__()
        self.fc1 = nn.Linear(x + y, h)
        self.fc21 = nn.Linear(h, z)
        self.fc22 = nn.Linear(h, z)
        self.fc3 = nn.Linear(z + y, h)
        self.fc4 = nn.Linear(h, x)
        self.loc = nn.Linear(y, z)
        self.scale = nn.Linear(y, z)
        self.classifier = Classifier(x, y, c)
        self.y = y
        self.x = x
        self.z = z
        self.h = h
        self.c = c
        self.sp = nn.Softplus()

    def classify(self, x):
        return self.classifier(x)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.sp(self.fc22(h1))

    def reparameterize(self, mu, var):
        pred_dist = dist.Independent(dist.Normal(mu, var), 1)
        self.pred_dist = pred_dist
        eps = pred_dist.rsample()
        prior_mean = self.loc(self.onehot)
        prior_std = self.sp(self.scale(self.onehot))
        self.prior = dist.Independent(dist.Normal(prior_mean, prior_std), 1)
        return eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def reconstruct(self, x, label):
        label_onehot = onehot_vector(label, self.y).type_as(x)
        self.onehot = label_onehot
        mu, var = self.encode(torch.cat([x, label_onehot], dim=1))
        z = self.reparameterize(mu, var)
        z = torch.cat([z, label_onehot], dim=1)
        return self.decode(z), {'mean': mu, 'var': var}

    def forward(self, x, label=None):
        supervised = (label is not None)
        x = x.view(-1, self.x)
        if not supervised:
            label = unsupervised_label(x, self.y)
            xs = x.repeat(self.y, 1)
        else:
            xs = x
        out, state = self.reconstruct(xs, label)
        reconstruction_loss = F.binary_cross_entropy(out, xs, reduction='none').sum(dim=-1)
        loss = reconstruction_loss
        # kl and prior
        kl_loss = dist.kl_divergence(self.pred_dist, self.prior)
        loss = loss + kl_loss
        prior_loss = torch.log2(torch.scalar_tensor(1/self.y)).to(kl_loss.device).type_as(kl_loss)
        loss = loss - prior_loss

        # classification
        if supervised:
            prob = self.classify(x) # [B, C]
            classification_loss = F.nll_loss(torch.log(prob), label, reduction=REDUCTION)
        else:
            classification_loss = 0
        loss_state = {'reconstruction': REDUCTION_METHOD(reconstruction_loss), 'kl': REDUCTION_METHOD(kl_loss), 'prior': prior_loss, 'classification': classification_loss}
        # if supervised, return current loss
        if supervised:
            return out, REDUCTION_METHOD(loss), classification_loss, loss_state, state
        # otherwise, compute the unsupervised loss
        prob = self.classify(x)
        loss = loss.view_as(prob.t()).t()

        # entropy
        H = torch.sum(prob * torch.log2(prob + 1e-8), dim=-1)
        L = torch.sum(prob * loss, dim=-1)

        loss = REDUCTION_METHOD(H+L)
        return out, loss, classification_loss, loss_state, state
