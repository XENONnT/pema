import matplotlib.pyplot as plt
import os


def save_canvas(name,
                save_dir='./figures',
                dpi=200,
                tight_layout=False,
                pickle_dump=False
                ):
    """Wrapper for saving current figure"""
    assert not pickle_dump
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/.')
    for sub_folder in 'pdf pkl svg'.split():
        sub_dir = os.path.join(save_dir, sub_folder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    if tight_layout:
        plt.tight_layout()
    if os.path.exists(save_dir):
        plt.savefig(f"{save_dir}/{name}.png", dpi=dpi, bbox_inches="tight")
        for extension in 'pdf svg'.split():
            plt.savefig(
                os.path.join(
                    save_dir,
                    extension,
                    f'{name}.{extension}'),
                dpi=dpi,
                bbox_inches="tight")
    else:
        raise FileExistsError(f'{save_dir} does not exist or does not have /pdf')

