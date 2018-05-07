#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os.path
import re
import argparse

matplotlib.rcParams.update({'font.size': 22})

PLOT_DIR = 'plots'


def extract_test_from_log(log_path, x_label):
  assert(x_label == "Step" or x_label == "Episode")
  reg = re.compile(f"^{x_label} \d*$") # Needed to avoid issues with other lines starting with 'Episode'

  with open(log_path) as log:
    lines = log.readlines()
    lines = list(filter(lambda x: '(test) Average reward' in x or reg.match(x.split(' - ')[1]) is not None, lines))
    # print(lines)
    
    indices = list(filter(lambda x: '(test) Average reward' in x[1], enumerate(lines)))
    x = map(lambda x: int(lines[x[0] - 1].split(' - ')[1].split(' ')[1].rstrip()), indices)
    y = map(lambda x: float(lines[x[0]].split(' - ')[1].split(': ')[1].rstrip()), indices)

    return list(x), list(y)

def extract_test_std_from_log(log_path, x_label):
  assert(x_label == "Step" or x_label == "Episode")
  reg = re.compile(f"^{x_label} \d*$") # Needed to avoid issues with other lines starting with 'Episode'

  with open(log_path) as log:
    lines = log.readlines()
    lines = list(filter(lambda x: 'Standard deviation' in x or reg.match(x.split(' - ')[1]) is not None, lines))
    # print(lines)
    
    indices = list(filter(lambda x: 'Standard deviation' in x[1], enumerate(lines)))
    x = map(lambda x: int(lines[x[0] - 1].split(' - ')[1].split(' ')[1].rstrip()), indices)
    y = map(lambda x: float(lines[x[0]].split(' - ')[1].split(': ')[1].rstrip()), indices)

    return list(x), list(y)

def extract_train_from_log(log_path, x_label):
  assert(x_label == "Step" or x_label == "Episode")
  reg = re.compile(f"^{x_label} \d*$")

  with open(log_path) as log:
    lines = log.readlines()
    lines = list(filter(lambda x: 'Episode reward mean' in x or reg.match(x.split(' - ')[1]) is not None, lines))
    # print(lines)
    indices = list(filter(lambda x: 'Episode reward mean' in x[1], enumerate(lines)))
    x = map(lambda x: int(lines[x[0] - 1].split(' - ')[1].split(' ')[1].rstrip()), indices)
    y = map(lambda x: float(lines[x[0]].split(' - ')[1].split(': ')[1].rstrip()), indices)

    return list(x), list(y)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Log Parser')
  parser.add_argument('log', type=str)
  parser.add_argument('--xmin', type=int, default=0)
  parser.add_argument('--xmax', type=int)
  parser.add_argument('--ymin', type=int, default=0)
  parser.add_argument('--ymax', type=int)
  parser.add_argument('--use-eps', action="store_true", default=False)
  parser.add_argument('--title', type=str)

  
  args = parser.parse_args()

  x_label = "Episode" if args.use_eps else "Step"
  tmp_name = '_'.join(list(map(lambda x: os.path.basename(x.strip()), args.log.split(','))))
  save_file_name = f'{PLOT_DIR}/{tmp_name}.png'
  
  filenames = args.log.split(',')

  test_x, test_y, train_x, train_y, test_error = ([], [], [], [], [])
  test_max = 0
  train_max = 0
  for name in filenames:
    name = name.strip()
    a, b = extract_test_from_log(name, x_label)
    test_x += list(map(lambda x: x + test_max, a))
    test_y += b
    a, b = extract_train_from_log(name, x_label)
    train_x += list(map(lambda x: x + train_max, a))
    train_y += b
    _, a = extract_test_std_from_log(name, x_label)
    test_error += a
    test_max = max(test_x)
    train_max = max(train_x)
  xmax = max(test_x + train_x)
  ymax = max(test_y + train_y)

  print(test_x)
  print(f"{list(filter(lambda x: x[1] >= 200, zip(test_x, test_y)))[0]} is first point of convergence")
  
  plt.figure(figsize=(16, 12), dpi=100)

  plt.fill_between(test_x, np.array(test_y) - np.array(test_error), 
    np.array(test_y) + np.array(test_error), color="red", alpha=0.2)
  test_plot = plt.plot(test_x, test_y, color="red", ms=25.0, label='100 episode average test reward')
  train_plot = plt.plot(train_x, train_y, color="black", ms=25.0, label='average train reward')
  plt.axvline(x=35000, linestyle='--')

  plt.xlim(xmin=args.xmin, xmax=xmax if args.xmax == None else args.xmax)
  plt.ylim(ymin=args.ymin, ymax=ymax if args.ymax == None else args.ymax)
  plt.xlabel(x_label)
  plt.ylabel('Average reward')
  plt.title(args.title if args.title != None else args.log)
  plt.legend()
  plt.savefig(save_file_name)
  plt.close()

  print(f"Saved to {save_file_name}")