import torch
import copy
from thop import profile

from network.starnet import starnet_s4


def calculate_recall(precision, f1, eps=1e-6):
    """
    根据给定的精确率（Precision）和F1分数（F1-Score）计算召回率（Recall）。

    参数:
        precision (float): 精确率，范围 [0, 1]
        f1 (float): F1分数，范围 [0, 1]
        eps (float): 浮点数计算的容差阈值，默认为1e-6

    返回:
        float: 召回率（Recall）

    异常:
        ValueError: 输入值不合法或无法计算有效的召回率
    """
    # 验证输入范围
    if not (0 <= precision <= 1) or not (0 <= f1 <= 1):
        raise ValueError("Precision and F1 must be in the range [0, 1]")

    recall=-precision/(1-2*precision/f1)
    return recall

if __name__ == "__main__":
    # device = "cuda"
    # backbone = starnet_s4(True).to(device)
    # test = backbone(torch.ones([1, 3, 128, 128]).float().to(device))
    #
    # for t in test:
    #     print(t.shape)
    #
    # n_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    # test_input = torch.randn(1, 3, 128, 128).float().to(device)  # 输入尺寸需匹配模型
    #
    # # 计算FLOPs和参数量
    # flops, _ = profile(copy.deepcopy(backbone), inputs=(test_input,))
    # gflops = flops / 1e9  # 转换为GFLOPs
    # print(f"FLOPs: {flops}")
    # print(f"GFLOPs: {gflops:.2f}")
    print(calculate_recall(0.9176,0.9377))