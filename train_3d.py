import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from einconv import Einconv
from train_fmnist import MLP

NUM_CLASSES = 10


def dummy_3d(x, add_dim, xp):
    return xp.repeat(x, add_dim).reshape(list(x.shape) + [add_dim])

# Network definition
class MLP_3D(MLP):
    def forward(self, x):
        x = dummy_3d(x, 28, self.xp)
        shape = [-1] + [1] + self.image_size
        x = F.reshape(x, shape)
        h1 = self.l1(x)
        h1 = F.max_pooling_3d(h1, ksize=2, stride=2)
        h2 = self.l2(h1)
        h2 = F.max_pooling_3d(h2, ksize=2, stride=2)
        h3 = self.l3(h2)
        return h3

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--channels', nargs='+', help='Numbers of input/output channels',
                        type=int, default=[64, 128])
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--graph', default='1_3_1_1_1_1_1_1')
    
    args = parser.parse_args()


    print(args)
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    shapes = {'channels': [1] + args.channels + [NUM_CLASSES],
              'image': [28, 28, 28],
              'filter': [args.filter_size] * 3,
              'inner_exp': 1,
              'batch': -1}
    
    model = L.Classifier(MLP_3D(args.graph, shapes))
    
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(weight_decay_rate=args.weight_decay)
    optimizer.setup(model)

    # Load the Fashion-MNIST dataset
    train, test = chainer.datasets.get_fashion_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
