import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def visualize_circular_sampling(H=5, W=5, center_i=2, center_j=2, r=1.5, num_points=8):
    """
    可视化单个点的圆形采样过程
    H, W: 特征图尺寸
    center_i, center_j: 中心点坐标
    r: 采样半径
    num_points: 采样点数
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    plt.rcParams.update({'font.size': 25, 'font.family': 'DejaVu Sans'})

    # 创建网格坐标系
    ax.set_xticks(np.arange(-0.5, W, 1))
    ax.set_yticks(np.arange(-0.5, H, 1))
    ax.grid(which='major', color='#CCCCCC', linestyle='-', linewidth=1)
    ax.set_xlim(-1, W)
    ax.set_ylim(-1, H)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 图像坐标系 (0,0)在左上角
    ax.set_title(f'Circular Sampling\nCenter: ({center_j},{center_i}), Radius: {r}, Points: {num_points}',
                 fontsize=16, pad=20)

    # 绘制网格点
    for i in range(H):
        for j in range(W):
            ax.plot(j, i, 'o', markersize=10, color='#1f77b4', alpha=0.7)

    # 标记中心点
    ax.plot(center_j, center_i, 'o', markersize=18, color='#ff7f0e', label=f'Center Point ({center_j},{center_i})')
    ax.text(center_j + 0.1, center_i + 0.1, 'Center', color='#d62728', fontsize=14, weight='bold')

    # 计算采样点
    angles = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
    sample_i = center_i + r * np.sin(angles)
    sample_j = center_j + r * np.cos(angles)

    # 绘制采样圆环
    circle = plt.Circle((center_j, center_i), r, fill=False,
                        linestyle='--', edgecolor='#2ca02c', linewidth=2, alpha=0.7)
    ax.add_patch(circle)

    # 绘制采样点和连接线
    for k, (sj, si) in enumerate(zip(sample_j, sample_i)):
        # 连接线
        ax.plot([center_j, sj], [center_i, si], '--', color='#9467bd', alpha=0.6)

        # 采样点
        ax.plot(sj, si, 's', markersize=14, color='#d62728', alpha=0.9)

        # 角度标注
        angle_deg = int(np.degrees(angles[k]))
        ax.text(sj + 0.25 * np.cos(angles[k]),
                si + 0.25 * np.sin(angles[k]),
                f'θ={angle_deg}°',
                color='#2ca02c', fontsize=12)

        # 点编号
        ax.text(sj + 0.05, si - 0.15, f'P{k}',
                color='white', weight='bold', fontsize=10,
                bbox=dict(facecolor='#d62728', alpha=0.8, boxstyle='round,pad=0.2'))

    # 添加图例和标注
    ax.text(0.02, 0.98, f'Sampling Parameters:\n- Radius: {r}\n- Points: {num_points}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='#f7f7f7', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig('circular_sampling_visualization.png', bbox_inches='tight', dpi=300)
    plt.show()


# 示例调用 (中心点2,2, 5x5网格)
visualize_circular_sampling(H=18, W=18, center_i=9, center_j=9, r=4, num_points=16)