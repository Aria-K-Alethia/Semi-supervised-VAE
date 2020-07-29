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
from dataset import get_mnist
from model import VAE, CVAE, StackedVAE, GMVAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1024, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--output', type=str, default='./model/model.pt')
parser.add_argument('--label', action='store_true', default=False)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('-architecture', type=str)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

labelled, unlabelled, validation = get_mnist(location="./data", batch_size=args.batch_size, labels_per_class=10)

prev_loss = float('inf')

model = HierVAE2().to(device)
x = 784
y = 10
z = 20
h = 400
c = [400, 128]
if args.architecture == 'vae':
    model = VAE(x, y, z, h)
elif args.architecture == 'cvae':
    model = CVAE(x, y, z, h, c)
elif args.architecture == 'stackedvae':
    vae = VAE(x, y, z, h)
    vae.load_state_dict(torch.load(args.pretrained_vae))
    model = StackedVAE(x, y, h, c, vae)
elif args.architecture == 'gmvae':
    model = GMVAE(x, y, z, h, c)
else:
    raise ValueError('Model architecture {} is not defined'.format(args.architecture))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


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
                y = onehot_vector(torch.randint(0, 10, (64,)), 10).to(device).type_as(sample)
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
            y = onehot_vector(torch.arange(10).repeat(2), 10).to(device).type_as(sample)
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
            y = onehot_vector(torch.arange(10), 10).to(device).type_as(sample)
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
            y = onehot_vector(torch.zeros(60).fill_(4), 10).to(device).type_as(sample)
            sample = torch.cat([sample, y], dim=1)
        sample = model.decode(sample).cpu()
        save_image(sample.view(60, 1, 28, 28), './output/traverse.png', nrow=6)

if __name__ == "__main__":
    if args.train:
        main_train()
    else:
        analysis()

