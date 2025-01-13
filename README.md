# PDF OCR 后端服务


## 必要的模型权重

```bash
# CRAFT
~/.EasyOCR/model/craft_mlt_25k.pth

# DBNet
~/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
~/.EasyOCR/model/pretrained_ic15_res18.pt
~/.EasyOCR/model/pretrained_ic15_res50.pt

# 文字识别
~/.EasyOCR/model/english_g2.pth
~/.EasyOCR/model/zh_sim_g2.pth
```


## 启动

```bash
python3.9 dispatcher.py 0
```
