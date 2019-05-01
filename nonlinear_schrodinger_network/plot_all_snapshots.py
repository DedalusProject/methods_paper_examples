"""
Plot output from NLS simulation.

Usage:
    graphplot.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import h5py
from graphplot import plot_graph


def plot_writes(filename, start, count, output, axes=None, save=True, dpi=100, amp_stretch=0.03, title=False):
    # Make axes if not provided
    if not axes:
        fig = plt.figure(figsize=(10,10))
        axes = fig.add_axes([0, 0, 1, 1])
    # Loop over assigned writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            plot_graph(file, index, axes, amp_stretch)
            # Remove axes
            axes.set_xlim(-1.1, 1.1)
            axes.set_ylim(-1.1, 1.1)
            axes.axis('off')
            # Timestamp title
            if title:
                axes.set_title('%.2f' %file['scales']['sim_time'][index], fontsize='large')
            # Save frame
            if save:
                savename = 'graph_%06i.png' %file['scales/write_number'][index]
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
            axes.cla()


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
    post.visit_writes(args['<files>'], plot_writes, output=output_path)

