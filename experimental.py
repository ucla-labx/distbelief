import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch

from torchvision import datasets, transforms

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True)


def parameter_server_test(model, log_df):
    from main import test
    test(model, 'cpu', test_loader, log_df)


def evaluate(logging_data):
    for i, row in enumerate(logging_data):
        row['iteration'] = i
    df = pd.DataFrame(logging_data)
    df.plot.line(x='iteration')
    plt.title('iteration v loss/acc rank {}'.format(dist.get_rank()))
    plt.savefig('process_{}.png'.format(get_rank()))
        
