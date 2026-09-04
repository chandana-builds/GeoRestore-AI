import torch


def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device
):
    """
    Validate model for one epoch.
    """

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for cloudy, target in dataloader:

            cloudy = cloudy.to(device)
            target = target.to(device)

            prediction = model(cloudy)

            loss = loss_fn(prediction, target)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss