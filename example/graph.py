"""
plots accuracy (test and train) vs. time
"""
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

colors = ['blue', 'green', 'red', 'orange', 'magenta']
files_to_read = ['log/single.csv', 'log/gpu.csv', 'log/node1.csv', 'log/node2.csv', 'log/node3.csv']
log_dataframes = list(map(pd.read_csv, files_to_read))

for df in log_dataframes:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] -= df['timestamp'].min()


def plot_train(df, label, color):
    plt.plot(df['timestamp'].dt.seconds / 3600.0,
             df['training_accuracy'].rolling(50).mean(),
             label=label,
             color=color)

def plot_test(df, label, color):
    plt.plot(df.dropna()['timestamp'].dt.seconds / 3600.0,
             df.dropna()['test_accuracy'].rolling(5).mean(),
             label=label,
             color=color)


fig1 = plt.figure(figsize=(20, 10))

for color, filename, df in zip(colors, files_to_read, log_dataframes):
    plot_train(df, filename, color)

plt.ylabel('Training Accuracy')
plt.xlabel('Time (hours)')
plt.legend()
plt.title("Training Accuracy vs. Time (50 iteration rolling average, freq: 3, lr: 0.1)")
plt.savefig('train_time.png')

fig = plt.figure(figsize=(20, 10))

for color, filename, df in zip(colors, files_to_read, log_dataframes):
    plot_test(df, filename, color)

plt.ylabel('Test Accuracy')
plt.xlabel('Time (hours)')
plt.legend()
plt.title("Test Accuracy vs. Time (5 iteration rolling average, freq: 3, lr: 0.1)")
plt.savefig('test_time.png')
