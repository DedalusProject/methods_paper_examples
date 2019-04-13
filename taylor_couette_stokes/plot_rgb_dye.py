"""
Plot RGB images using three dye fields

Usage:
    plot_rgb_dye.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""
from dedalus.extras import plot_tools
import numpy as np
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools
import h5py

def trim_zero(data):
    data[data<0] = 0
    return data

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['c1', 'c2', 'c3']
    scale = 2.5
    dpi = 100
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    fig = plt.figure()
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        r = file['scales/r/10']
        theta = file['scales/θ/10']

        theta,r = plot_tools.quad_mesh(theta,r)
        c1 = file['tasks/c1']
        c2 = file['tasks/c2']
        c3 = file['tasks/c3']

        for index in range(start, start+count):
            c1_mod = c1[index]
            c1_mod[c1[index]<0] = 0
            c1_mod[c1[index]>1] = 1
            c2_mod = c2[index]
            c2_mod[c2[index]<0] = 0
            c2_mod[c2[index]>1] = 1
            c3_mod = c3[index]
            c3_mod[c3[index]<0] = 0
            c3_mod[c3[index]>1] = 1
            colors = tuple(np.array([c1_mod.T.flatten(),c2_mod.T.flatten(),c3_mod.T.flatten()]).transpose().tolist())
            ax = fig.add_subplot(111,projection='polar')
            ax.pcolormesh(theta,r,c1[0,:,:].T,color=colors)
            ax.spines['polar'].set_visible(False)

            ## removing the tick marks
            ax.set_xticks([])
            ax.set_yticks([])

            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

