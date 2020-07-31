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
from utils import onehot_vector
from itertools import cycle

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
parser.add_argument('--architecture', type=str)
parser.add_argument('--pretrained-vae', type=str, default='./model/vae.pt')
parser.add_argument('--labels-per-class', type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

labelled, unlabelled, validation = get_mnist(location="./data", batch_size=args.batch_size, labels_per_class=args.labels_per_class)

prev_loss = float('inf')

X = 784
Y = 10
Z = 20
H = 400
C = [400, 128]
if args.architecture == 'vae':
    model = VAE(X, Y, Z, H)
elif args.architecture == 'cvae':
    model = CVAE(X, Y, Z, H, C)
elif args.architecture == 'stackedvae':
    vae = VAE(X, Y, Z, H)
    vae.load_state_dict(torch.load(args.pretrained_vae))
    model = StackedVAE(X, Y, Z, H, C, vae)
elif args.architecture == 'gmvae':
    model = GMVAE(X, Y, Z, H, C)
else:
    raise ValueError('Model architecture {} is not defined'.format(args.architecture))
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    print('Train start, labelled: {}, unlablled: {}'.format(len(labelled), len(unlabelled)))
    if epoch == 1:
        for x, y in labelled:
            continue
        for x, y in unlabelled:
            continue
    for batch_idx, ((x, y), (u, _)) in enumerate(zip(cycle(labelled), unlabelled)):
    #for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
        x = x.to(device)
        y = y.to(device)
        u = u.to(device)
        optimizer.zero_grad()
        # labelled data
        l_recon_batch, L, classification_loss, l_loss_state, l_state = model(x, y)
        u_recon_batch, U, _, u_loss_state, u_state = model(u)
        if args.architecture == 'vae':
            loss = U
        else:
            loss = L + U + args.alpha * classification_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, L_BCE: {:.6f}, L_KLD: {:.6f}, L_CLAS: {:.6f}, U_BCE: {:.6f}, U_KLD: {:.6f}'.format(
                epoch, batch_idx * len(x), len(unlabelled.dataset),
                100. * batch_idx / len(unlabelled),
                loss.item() / len(x),
                l_loss_state['reconstruction'].item() / len(x),
                l_loss_state['kl'].item() / len(x),
                l_loss_state['classification'].item() / len(x),
                u_loss_state['reconstruction'].item() / len(x),
                u_loss_state['kl'].item() / len(x)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(unlabelled.dataset)))

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
        for i, (x, y) in enumerate(validation):
            x = x.to(device)
            y = y.to(device)
            recon_batch, loss, classification_loss, loss_state, state = model(x, y)
            test_loss += loss_state['reconstruction'].item()
            if i == 0:
                n = min(x.shape[0], 8)
                comparison = torch.cat([x.view(x.shape[0], 1, 28, 28)[:n],
                                      recon_batch.view(x.shape[0], 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/{}_reconstruction_'.format(args.architecture) + str(epoch) + '.png', nrow=n)

    global prev_loss
    test_loss /= len(validation.dataset)
    if test_loss > prev_loss:
        lr_schedule()
    prev_loss = test_loss
    print('====> Test set loss: {:.4f}'.format(test_loss))

def random_sample(epoch):
    with torch.no_grad():
            if args.architecture != 'gmvae':
                sample = torch.randn(64, 20).to(device)
                y = onehot_vector(torch.randint(0, 10, (64,)), 10).to(device).type_as(sample)
            else:
                y = onehot_vector(torch.randint(0, 10, (64,)), 10).to(device).float()
                loc = model.loc(y)
                scale = model.sp(model.scale(y))
                temp_dist = dist.Independent(dist.Normal(loc, scale), 1)
                sample = temp_dist.rsample()
            if args.label:
                sample = torch.cat([sample, y], dim=1)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/{}_sample_'.format(args.architecture) + str(epoch) + '.png')

def main_train():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        random_sample(epoch)
    state_dict = model.state_dict()
    torch.save(state_dict, args.output)

def analysis():
    state_dict = torch.load(args.output)
    model.load_state_dict(state_dict)
    embedding, label = None, None
    # latent variable visualization and unsupervised accuracy
    plt.figure()
    tsne = TSNE(2, 50, init='pca')
    #tsne = PCA(n_components=2, whiten=True)
    correct_count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(validation):
            x = x.to(device)
            y = y.to(device)
            recon_batch, _, _, _, state= model(x, y)
            mu = state['mean']
            if embedding is None:
                embedding = state['mean']
                label = y
            else:
                embedding = torch.cat([embedding, mu], 0)
                label = torch.cat([label, y], 0)
            if args.architecture == 'stackedvae':
                feat = model.vae.sample(x)
                logits = model.classify(feat)
            else:
                logits = model.classify(x)
            temp = torch.argmax(logits, dim=-1)
            correct_count += (torch.argmax(logits, dim=-1).squeeze() == y).sum().item()
    accuracy = correct_count / len(validation.dataset)
    print('Unsupervised accuracy: {:.2f}%'.format(accuracy * 100))
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
    f.savefig('./output/{}_{}_latent_variable.png'.format(args.architecture, args.labels_per_class))
    plt.clf()
    '''
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
    '''
    buf = [-3, -1.5, 0, 1.5, 3]
    if args.architecture != 'gmvae':
        base_sample = torch.zeros(Z).to(device)
        sample = []
        with torch.no_grad():
            for i in range(Z * len(buf)):
                temp = base_sample.clone()
                temp[i//len(buf)] = buf[i%len(buf)]
                sample.append(temp)
            sample = torch.stack(sample)
            if args.label:
                y = onehot_vector(torch.cat([torch.ones(Z * len(buf) // Y) * i for i in range(Y)]), Y).to(device).type_as(sample)
                sample = torch.cat([sample, y], dim=1)
            sample = model.decode(sample).cpu()
    else:
        y = onehot_vector(torch.cat([torch.ones(Z * len(buf) // Y) * i for i in range(Y)]), Y).to(device).float()
        mean = model.loc(y)
        scale = model.scale(y)
        for i in range(y.shape[0]):
            dim = i // len(buf)
            index = i % len(buf)
            mean[i, dim] = mean[i, dim] + buf[index] * scale[i, dim]
        sample = torch.cat([mean, y], dim=1)
        sample=  model.decode(sample).cpu()
    save_image(sample.view(Z * len(buf), 1, 28, 28), './output/{}_traverse.png'.format(args.architecture), nrow=2 * len(buf))

if __name__ == "__main__":
    if args.train:
        main_train()
    else:
        analysis()

