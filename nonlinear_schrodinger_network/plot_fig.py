"""
Plot output from NLS simulation.

Usage:
    graphplot.py <file> [--output=<dir>]

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


def plot_panel(filename):
    # Parameters
    panels = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    cols = 3
    amp_stretch = 0.03
    title = False

    # Make figure
    N = len(panels)
    J = cols
    I = int(np.ceil(N/J))
    fig = plt.figure(figsize=(J, I))
    # Loop over panels
    with h5py.File(filename, mode='r') as file:
        for n, index in enumerate(panels):
            i, j = divmod(n, J)
            i = I - i - 1
            axes = fig.add_axes([j/J, i/I, 1/J, 1/I])
            plot_graph(file, index, axes, amp_stretch, lw=0, fc='k', alpha=0.75)
            # Remove axes
            axes.set_xlim(-1.1, 1.1)
            axes.set_ylim(-1.1, 1.1)
            axes.axis('off')
            # Timestamp title
            if title:
                axes.set_title('%.2f' %file['scales']['sim_time'][index], fontsize='large')
    # Save frame
    fig.savefig('fig_nls.pdf' )
    fig.savefig('fig_nls.jpg', dpi=400)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    plot_panel(args['<file>'])

