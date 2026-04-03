import torch
from model import DepthNet, PoseNet
from loss import photometric_loss, smoothness_loss

device = "cpu"

depth_net = DepthNet().to(device)
pose_net = PoseNet().to(device)

optimizer = torch.optim.Adam(
    list(depth_net.parameters()) + list(pose_net.parameters()), lr=1e-4
)

for epoch in range(2):
    img1 = torch.randn(1, 3, 64, 64).to(device)
    img2 = torch.randn(1, 3, 64, 64).to(device)

    depth = depth_net(img1)
    pose = pose_net(torch.cat([img1, img2], dim=1))

    loss1 = photometric_loss(img1, img2)
    loss2 = smoothness_loss(depth)

    loss = loss1 + 0.1 * loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")