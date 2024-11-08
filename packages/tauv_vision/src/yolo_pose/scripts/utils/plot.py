import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from math import sqrt, ceil
from typing import Optional
import torch.nn.functional as F
import pathlib


def save_plot(fig: plt.Figure, save_dir: Optional[pathlib.Path], name: str):
    if save_dir is not None:
        save_path = save_dir / name
        fig.savefig(save_path)
    else:
        fig.show()


def plot_prototype(prototype: torch.Tensor) -> plt.Figure:
    # protoype is n_prototypes x h x w
    fig = plt.figure()

    depth = prototype.size(0)
    nrows = int(ceil(sqrt(depth)))

    grid = ImageGrid(fig,111, nrows_ncols=(nrows, nrows), share_all=True, cbar_mode="single", axes_pad=0.2, cbar_pad=0.5)

    for i in range(depth):
        img = grid[i].imshow(prototype[i].detach().cpu())

    grid.cbar_axes[0].colorbar(img)

    return fig


def plot_belief(belief: torch.Tensor, img: Optional[torch.Tensor] = None, points: Optional[torch.Tensor] = None, truth_points: Optional[torch.Tensor] = None):
    # belief is belief_depthxhxw
    # img is 3xHxW

    belief_depth = belief.size(0)
    belief_size = belief.size()[1:3]


    # overlay is 3xhxw
    if img is not None:
        img_resized = 0.5 * F.interpolate(img.unsqueeze(0), belief_size, mode="bilinear").squeeze(0)  # 3xhxw

        overlay = (belief.unsqueeze(1) * img_resized.unsqueeze(0)) + img_resized.unsqueeze(0)
    else:
        overlay = belief

    overlay = torch.clamp(overlay, 0, 1)

    fig = plt.figure()

    nrows = int(ceil(sqrt(belief_depth)))
    grid = ImageGrid(fig,111, nrows_ncols=(nrows, nrows), share_all=True, cbar_mode="single", axes_pad=0.2, cbar_pad=0.5)

    for i in range(belief_depth):
        if len(overlay.size()) == 4:
            im = grid[i].imshow(overlay[i].permute(1, 2, 0).detach().cpu())
        else:
            im = grid[i].imshow(overlay[i].detach().cpu())

    grid.cbar_axes[0].colorbar(im)

    return fig

if __name__ == "__main__":
    # prototype_fig = plot_prototype(torch.rand((52, 32, 32)))
    # plt.show()

    belief = torch.rand((9, 32, 32))
    img = torch.rand((3, 64, 64))
    points = torch.randint(0, 32, (9, 2))
    truth_points = torch.randint(0, 32, (9, 2))

    belief_fig = plot_belief(belief, img, points, truth_points)
    plt.show()

    belief_fig = plot_belief(belief, None, points, truth_points)
    plt.show()
