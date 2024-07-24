import EmbeddingNet
import Triplet_loss
from Data import train_loader
import torch.optim as optim


model = EmbeddingNet()
loss = Triplet_loss()
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
epoch_losses = []
for epoch in range(2):
    batch_losses = []
    for batch_idx, batch in enumerate(train_loader):
        anchors = batch["anchor"]
        positives = batch["positive"]
        negatives = batch["negative"]

        anchor_embeddings = model(anchors)

        positive_embeddings = model(positives)

        negative_embeddings = model(negatives)

        loss_value = loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        batch_losses.append(loss_value.item())
        print(
            f"Epoch {epoch+1}/2, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss_value.item()}"
        )
    epoch_loss = sum(batch_losses) / len(
        batch_losses
    )  # Calculate average loss for the epoch
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/2, Loss: {epoch_loss}")
