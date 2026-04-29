import torch
import torch.nn as nn

# =========================
# ✅ 极简 Action Head（替代官方）
# =========================
class TinyActionHead(nn.Module):
    def __init__(self, input_dim=128, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, batch):
        x = batch["vlm_output"].mean(dim=1)  # (B, D)
        pred = self.net(x)

        gt = batch["action"]
        loss = ((pred - gt) ** 2).mean()

        return {"action_loss": loss}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 device:", device)

    model = TinyActionHead().to(device)
    model.train()

    # =========================
    # fake VLM
    # =========================
    B, T, D = 2, 1, 128
    fake_vlm = torch.randn(B, T, D, device=device)

    # =========================
    # fake action
    # =========================
    fake_action = torch.randn(B, 16, device=device)

    batch = {
        "vlm_output": fake_vlm,
        "action": fake_action
    }

    # =========================
    # forward
    # =========================
    print("🧠 forward...")
    out = model(batch)

    loss = out["action_loss"]
    print("✅ loss:", loss.item())

    # =========================
    # backward
    # =========================
    loss.backward()
    print("🔥 backward OK")


if __name__ == "__main__":
    main()