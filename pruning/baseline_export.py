import torch
from yunet_standalone import YuNet

model = YuNet()
model.load_pretrained(
      "/training/libfacedetection.train/weights/yunet_n.pth"
  )

 # Optional: also set up an optimizer if you want it in the checkpoint
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


#This is so I can load that in ckpt format if I wanted to
torch.save({
      "epoch": 0,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": float("inf"),
      "model_cfg": "yunet_n",   # so future you knows which config to instantiate
  }, "yunet_n_baseline.ckpt")