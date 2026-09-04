import torch.optim as optim


def get_optimizer(model, learning_rate=1e-4):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    return optimizer 
