import numpy as np

# set up various things for postscript eps plotting.
golden_mean = (np.sqrt(5)-1.0)/2.0
fig_width = 3.4 # in column width
fig_height = fig_width * golden_mean
fig_size =  [fig_width,fig_height]

params = {
          'axes.labelsize': 10,
          'font.family':'CMU Serif',
          #'text.fontsize': 8,
          'font.size':8,
          'font.serif': 'CMU Serif Roman',
          'legend.fontsize': 8,
          #'legend.markersize': 2,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
#          'text.usetex': True,
          'figure.figsize': fig_size,
          'figure.dpi': 600,
          'lines.markersize':3,
          'lines.linewidth': 1,
          #'lines.dashes':(),
          'figure.subplot.left': 0.20,
          'figure.subplot.bottom': 0.20,
	  'figure.subplot.right': 0.95,
	  'figure.subplot.top': 0.90,
	  'figure.subplot.hspace': 0.1
          }

    
#def setup_plot():
#
#    import publication_settings
#
#    rcParams.update(publication_settings.params)
