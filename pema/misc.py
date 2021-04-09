import matplotlib.pyplot as plt
import os


def save_canvas(name, save_dir='./figures', tight_layout=False):
    """Wrapper for saving current figure"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/.')
    if not os.path.exists(save_dir + '/pdf/.'):
        os.makedirs(save_dir + '/pdf/.')
    if not os.path.exists(save_dir + '/svg/.'):
        os.makedirs(save_dir + '/svg/.')
    if tight_layout:
        plt.tight_layout()
    if os.path.exists(save_dir) and os.path.exists(save_dir + '/pdf'):
        plt.savefig(f"{save_dir}/{name}.png", dpi=200, bbox_inches="tight")
        plt.savefig(f"{save_dir}/pdf/{name}.pdf", dpi=100, bbox_inches="tight")
        plt.savefig(f"{save_dir}/svg/{name}.svg", dpi=100, bbox_inches="tight")
    else:
        raise FileExistsError(f'{save_dir} does not exist or does not have /pdf')
