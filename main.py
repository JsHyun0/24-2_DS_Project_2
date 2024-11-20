import model
import argparse
import torch

import util


args = util.parse_args()
with open('args.data', 'r') as f:
