import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='Name of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# Add more arguments as needed

opt = parser.parse_args()
print(opt)