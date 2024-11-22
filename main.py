import model
import argparse
import torch
import util
import torch.nn as nn
import torch.nn.functional as F
import args

args = args.parse_args()
train_data, valid_data, cat_features, num_features = util.process_data(args.data_path)

model = model.myModel(args.encoding_dim, cat_features, num_features, args.num_classes, args.dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    for batch in train_data:
        x_cat, x_num, y = batch
        y_pred = model(x_cat, x_num)
        loss = F.cross_entropy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
