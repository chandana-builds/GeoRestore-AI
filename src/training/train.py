import torch


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scheduler=None):
    """
    Train model for one epoch with gradient clipping and learning rate scheduling.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 25 == 0 or (step + 1) == total_batches:
            avg_so_far = running_loss / (step + 1)
            print(f"  Step [{step + 1}/{total_batches}] - Batch Loss: {loss.item():.4f} - Avg Loss: {avg_so_far:.4f}", flush=True)


    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / total_batches
    return epoch_loss