import neptune
import optuna
import torch
import torch.nn as nn
import model
import util
import args
import os




def objective(trial):
    run = neptune.init_run(
        project="jshyun/Datascience-Final",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )  # your credentials

    args = args.parse_args()
    data = util.process_data(args.data_path)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
    run["parameters"] = params

    model = model.myModel(args.encoding_dim, args.cat_features, args.num_features, args.num_classes, args.dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_loguniform('lr', 1e-4, 1e-2))
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        for batch in data:
            x_cat, x_num, y = batch
            y_pred = model(x_cat, x_num)
            loss = criterion(y_pred, y)

    run.stop()
    return loss.item()

if __name__ == "__main__":
    optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)