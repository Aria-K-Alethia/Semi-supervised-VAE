from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributions as dist
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--output', type=str, default='./model/model.pt')
parser.add_argument('--label', action='store_true', default=False)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

prev_loss = float('inf')

def onehot(label, number):
    # label: [#batch_size]
    # number: label number
    device = label.device
    out = torch.arange(number).to(device)
    out.unsqueeze_(0)
    batch_size = label.shape[0]
    out = out.repeat(batch_size, 1)
    label = label.unsqueeze(1)
    temp = (out != label)
    out[out == label] = 1
    out[temp] = 0
    return out

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.loc = nn.Parameter(torch.zeros(20), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(20), requires_grad=False)
        self.prior = dist.Independent(dist.Normal(self.loc, self.scale), 1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        pred_dist = dist.Independent(dist.Normal(mu, std), 1)
        eps = pred_dist.rsample()
        kl_loss = dist.kl_divergence(pred_dist, self.prior)
        return eps, kl_loss

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, label):
        mu, logvar = self.encode(x.view(-1, 784))
        z, kl_loss = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, kl_loss

class StackVAE(nn.Module):
    def __init__(self):
        super(StackVAE, self).__init__()
        self.fc1 = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(30, 400)
        self.fc4 = nn.Linear(400, 784)
        self.loc = nn.Parameter(torch.zeros(20), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(20), requires_grad=False)
        self.prior = dist.Independent(dist.Normal(self.loc, self.scale), 1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        pred_dist = dist.Independent(dist.Normal(mu, std), 1)
        eps = pred_dist.rsample()
        kl_loss = dist.kl_divergence(pred_dist, self.prior)
        kl_loss -= torch.log(torch.scalar_tensor(1/10)).to(kl_loss.device).type_as(kl_loss)
        return eps, kl_loss

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, label):
        label_onehot = onehot(label, 10).type_as(x)
        mu, logvar = self.encode(torch.cat([x.view(-1, 784), label_onehot], dim=1))
        z, kl_loss = self.reparameterize(mu, logvar)
        z = torch.cat([z, label_onehot], dim=1)
        return self.decode(z), mu, logvar, kl_loss

class GMVAE(nn.Module):
    def __init__(self):
        super(GMVAE, self).__init__()
        self.fc1 = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        #self.loc = nn.Parameter(torch.zeros(20), requires_grad=False)
        #self.scale = nn.Parameter(torch.ones(20), requires_grad=False)
        #self.prior = dist.Independent(dist.Normal(self.loc, self.scale), 1)
        self.loc = nn.Linear(10, 20, bias=False)
        self.scale = nn.Linear(10, 20)
        self.sp = nn.Softplus()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        pred_dist = dist.Independent(dist.Normal(mu, std), 1)
        eps = pred_dist.rsample()
        prior_mean = self.loc(self.onehot)
        prior_std = self.sp(self.scale(self.onehot))
        self.prior = dist.Independent(dist.Normal(prior_mean, prior_std), 1)
        kl_loss = dist.kl_divergence(pred_dist, self.prior)
        kl_loss -= torch.log(torch.scalar_tensor(1/10)).to(kl_loss.device).type_as(kl_loss)
        return eps, kl_loss

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, label):
        label_onehot = onehot(label, 10).type_as(x)
        self.onehot = label_onehot
        mu, logvar = self.encode(torch.cat([x.view(-1, 784), label_onehot], dim=1))
        z, kl_loss = self.reparameterize(mu, logvar)
        #z = torch.cat([z, label_onehot], dim=1)
        return self.decode(z), mu, logvar, kl_loss


model = HierVAE2().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_loss):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.sum(kl_loss)
    #KLD2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print(KLD2, KLD)
    return BCE + KLD, (BCE, KLD)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, kl_loss = model(data, label)
        loss, (BCE, KLD) = loss_function(recon_batch, data, mu, logvar, kl_loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, BCE: {:.6f}, KLD: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                BCE.item() / len(data), KLD.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def lr_schedule():
    old_lr = optimizer.param_groups[0]['lr']
    factor = 0.4
    lr = old_lr * factor
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    print('Learning rate change: {} -> {}'.format(old_lr, lr))

def test(epoch):
    model.eval()
    test_loss = 0
    outdata = []
    with torch.no_grad():
        for i, (data, y) in enumerate(test_loader):
            data = data.to(device)
            y = y.to(device)
            recon_batch, mu, logvar, kl_loss = model(data, y)
            loss, (BCE, KLD) = loss_function(recon_batch, data, mu, logvar, kl_loss)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            
    global prev_loss
    test_loss /= len(test_loader.dataset)
    if test_loss > prev_loss:
        lr_schedule()
    prev_loss = test_loss
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main_train():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            if args.label:
                y = onehot(torch.randint(0, 10, (64,)), 10).to(device).type_as(sample)
                sample = torch.cat([sample, y], dim=1)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
    state_dict = model.state_dict()
    torch.save(state_dict, args.output)

def analysis():
    state_dict = torch.load(args.output)
    model.load_state_dict(state_dict)
    embedding, label = None, None
    plt.figure()
    tsne = TSNE(2, 10, init='pca')
    #pca = PCA(n_components=2, whiten=True)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            recon_batch, mu, logvar, kl_loss = model(x, y)
            if embedding is None:
                embedding = mu
                label = y
            else:
                embedding = torch.cat([embedding, mu], 0)
                label = torch.cat([label, y], 0)
    embedding2 = embedding.cpu().numpy()
    label = label.cpu().numpy()
    label = label[:10000]
    embedding2 = embedding2[:10000]
    #pca.fit(embedding)
    #out = pca.transform(embedding)
    #print(pca.explained_variance_ratio_)
    out = tsne.fit_transform(embedding2)
    out = (out - out.min(0)) / (out.max(0) - out.min(0))
    for i in range(10):
        d = out[label==i]
        plt.scatter(d[:, 0], d[:, 1], label=str(i))
    plt.legend(loc='upper right')
    f = plt.gcf()
    f.savefig('./output/latent_variable.png')
    plt.clf()
    with torch.no_grad():
        sample = torch.diag(torch.ones(20)).to(device)
        if args.label:
            y = onehot(torch.arange(10).repeat(2), 10).to(device).type_as(sample)
            sample = torch.cat([sample, y], dim=1)
        sample = model.decode(sample).cpu()
        save_image(sample.view(20, 1, 28, 28), './output/sample.png')
    buf = []
    for i in range(10):
        mu = embedding[label==i]
        mu = torch.mean(mu, 0)
        buf.append(mu)
    sample = torch.stack(buf)
    with torch.no_grad():
        if args.label:
            y = onehot(torch.arange(10), 10).to(device).type_as(sample)
            sample = torch.cat([sample, y], dim=1)
        sample = model.decode(sample).cpu()
        save_image(sample.view(10, 1, 28, 28), './output/mean.png')
    base_sample = torch.zeros(20).to(device)
    buf = [-2, 0, 2]
    sample = []
    with torch.no_grad():
        for i in range(60):
            temp = base_sample.clone()
            temp[i//3] = buf[i%3]
            sample.append(temp)
        sample = torch.stack(sample)
        if args.label:
            y = onehot(torch.zeros(60).fill_(4), 10).to(device).type_as(sample)
            sample = torch.cat([sample, y], dim=1)
        sample = model.decode(sample).cpu()
        save_image(sample.view(60, 1, 28, 28), './output/traverse.png', nrow=6)
        
if __name__ == "__main__":
    if args.train:
        main_train()
    else:
        analysis()
    
