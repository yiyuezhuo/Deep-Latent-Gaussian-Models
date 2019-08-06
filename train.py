
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from itertools import chain
import os

#from generative import GenerativeModel
#from recognition import RecognitionModel
import generative
import recognition
from losses import loss_function

import cholesky_factor

parser = argparse.ArgumentParser(description='DLGM MNIST training')
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
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                    help='learning rate')
parser.add_argument('--output-dir', default="results")
parser.add_argument('--generative-model', default = 'GenerativeModel')
parser.add_argument('--recognition-model', default = 'RecognitionModel')
parser.add_argument('--chol-factor-cls', default='CholeskyFactor')
parser.add_argument('--checkpoints-dir', default = 'checkpoints')
parser.add_argument('--checkpoint-tag', default='')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../dataset', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../dataset', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

chol_factor_cls = cholesky_factor.__dict__[args.chol_factor_cls]
#generative_model = GenerativeModel().to(device)
#recognition_model = RecognitionModel(chol_factor_cls = chol_factor_cls).to(device)
generative_model = generative.__dict__[args.generative_model]().to(device)
recognition_model = recognition.__dict__[args.recognition_model](chol_factor_cls = chol_factor_cls).to(device)

optimizer = optim.Adam(chain(generative_model.parameters(), recognition_model.parameters()), lr=args.lr)

def train(epoch):
    #model.train()
    generative_model.train()
    recognition_model.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        #recon_batch, mu, logvar = model(data)
        mu, R = recognition_model(data.view(-1, 28*28))
        z = recognition_model.sample(mu, R)
        recon_batch = generative_model(z)

        loss = loss_function(recon_batch, data, mu, R)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    generative_model.eval()
    recognition_model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            #recon_batch, mu, logvar = model(data)
            mu, R = recognition_model(data.view(-1, 28*28))
            z = recognition_model.sample(mu, R)
            recon_batch = generative_model(z)

            test_loss += loss_function(recon_batch, data, mu, R).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         args.output_dir+'/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    #os.makedirs('results', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            #sample = torch.randn(64, 20).to(device)
            sample = generative_model.sample_prior(64, device = device)
            #sample = model.decode(sample).cpu()
            sample = generative_model(sample)

            save_image(sample.view(64, 1, 28, 28),
                       args.output_dir+'/sample_' + str(epoch) + '.png')

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    name = '{}_{}_{}{}.pth'.format(
        args.generative_model,
        args.recognition_model,
        args.chol_factor_cls,
        args.checkpoint_tag)
    path = os.path.join(args.checkpoints_dir, name)

    checkpoint = {}
    checkpoint['generative_model'] = generative_model.state_dict()
    checkpoint['recognition_model'] = recognition_model.state_dict()
    torch.save(checkpoint, path)


'''
python train.py --lr 1e-4
python train.py --lr 1e-4 --generative-model GenerativeMNIST --recognition-model RecognitionMNIST
ipython -i train.py -- --lr 1e-4 --generative-model GenerativeMNISTLarge --recognition-model RecognitionMNIST --output-dir results_large
ipython -i train.py -- --lr 1e-4 --generative-model GenerativeMNISTLarge --recognition-model RecognitionMNIST --output-dir results_large --epochs 100
ipython -i train.py -- --lr 1e-4 --generative-model GenerativeMNISTVAE --recognition-model RecognitionMNISTVAE --chol-factor-cls DiagonalFactor --output-dir results_vae_like
'''