import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1.0)])


train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

print(f"No of training samples : {len(train_dataset)}")
print(f"No of test samples : {len(test_dataset)}")


class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.data
        self.labels = dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor = self.data[index]
        anchor_label = self.labels[index]

        # Sample a positive sample from the same class
        positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0])
        while positive_idx == index:
            positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0])
        positive = self.data[positive_idx]

        # Sample a negative sample from a different class
        negative_idx = np.random.choice(np.where(self.labels != anchor_label)[0])
        negative = self.data[negative_idx]

        return {"anchor": anchor, "positive": positive, "negative": negative}


batch_size = 512
train_triplet_dataset = TripletDataset(train_dataset)
train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, shuffle=True)

for batch in train_loader:
    anchors = batch["anchor"]
    positives = batch["positive"]
    negatives = batch["negative"]
    print(anchors.shape)
    print(positives.shape)
    print(negatives.shape)
    fig, axes = plt.subplots(1, 3)  # Create a figure with 3 subplots

    # Display the anchor image
    axes[0].imshow(anchors[0].squeeze(), cmap="gray")
    axes[0].set_title("Anchor")
    axes[0].axis("off")  # Turn off axis labels

    # Display the positive image
    axes[1].imshow(positives[0].squeeze(), cmap="gray")
    axes[1].set_title("Positive")
    axes[1].axis("off")  # Turn off axis labels

    # Display the negative image
    axes[2].imshow(negatives[0].squeeze(), cmap="gray")
    axes[2].set_title("Negative")
    axes[2].axis("off")  # Turn off axis labels
    break
