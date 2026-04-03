import torch
import os
import matplotlib.pyplot as plt
from model import DepthNet, PoseNet
from loss import photometric_loss, smoothness_loss

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

device = "cpu"

depth_net = DepthNet().to(device)
pose_net = PoseNet().to(device)

optimizer = torch.optim.Adam(
    list(depth_net.parameters()) + list(pose_net.parameters()), lr=1e-4
)

# Clear previous loss file (optional)
open("results/loss.txt", "w").close()

for epoch in range(2):
    # Toy dataset (random images)
    img1 = torch.randn(1, 3, 64, 64).to(device)
    img2 = torch.randn(1, 3, 64, 64).to(device)

    # Forward pass
    depth = depth_net(img1)
    pose = pose_net(torch.cat([img1, img2], dim=1))

    # Loss
    loss1 = photometric_loss(img1, img2)
    loss2 = smoothness_loss(depth)
    loss = loss1 + 0.1 * loss2

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save loss to file
    with open("results/loss.txt", "a") as f:
        f.write(f"Epoch {epoch}, Loss: {loss.item()}\n")

    # Save depth image at last epoch
    if epoch == 1:
        plt.imshow(depth.detach().cpu().squeeze(), cmap='plasma')
        plt.title("Predicted Depth Map (Toy Data)")
        plt.colorbar()
        plt.savefig("results/depth.png")
        plt.close()