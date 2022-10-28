import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.font_manager as font_manager
import matplotlib
from matplotlib.transforms import Bbox
import numpy as np
import pandas


'''
Possible pacakages for the function to work:

python     3.7.10
matplotlib 3.4.2

!sudo apt install texlive-fonts-recommended texlive-fonts-extra
!sudo apt install dvipng
!sudo apt install texlive-full
!sudo apt install texmaker
!sudo apt install ghostscript

Parameters:
--------------
text_list = [{'x':-6.55,'y': -12.0, 'text':'(a) FID($\downarrow$)', 'fontsize': 16},
             {'x':-2.945,'y': -12.0, 'text':'(b) class-FID($\downarrow$)', 'fontsize': 16},
            ]
            
bbox_inches = Bbox(np.array([[-0.22, -0.7], [12.9, 1.93]]))

read_fn 
function used to read a list of csv files
returns a list of dictionaries to be processed by plot_fn

plot_fn
function used to plot figures given a dictionary of hyperparameters and an axis
no returns


subplots
[0,0]             [0,1]  ... [0, num_axis[1]]
[1,0]
.
.
.
[num_axis[0], 0]         ...
--------------
'''

class Reader:
  ###=========== Read csv files to produce publication quality figures ==============
  @staticmethod
  def normal_read_fn(csv_path_list):
      plots_dict_list = []
      for i in range(len(csv_path_list)):
          plots_dict = {}
          
          path = csv_path_list[i]
          raw  = pandas.read_csv(path, header=0)
          df   = pandas.DataFrame(raw)
           
          plots_dict['names']       = list(df.columns)[1:]
          plots_dict['df']          = [pandas.DataFrame(df[name]) for name in plots_dict['names']]
          for d in plots_dict['df']: d.columns = ['y']
          plots_dict['xlabel']      = None
          plots_dict['ylabel']      = None 
          plots_dict['y_lim']       = None 
          plots_dict['yticks']      = None 
          plots_dict['xticks']      = None
          plots_dict['yticklabels'] = None
          plots_dict['xticklabels'] = None
          plots_dict['alpha']       = 0.25
          plots_dict['smooth']      = False
          plots_dict['sigma']       = False
          plots_dict['markon']      = False
          
          plots_dict_list.append(plots_dict) 
      #=== special settings for each plots === 
      # plots_dict_list[1]['names'] = [name.replace('_', '\_') for name in plots_dict_list[1]['names']]
      # plots_dict_list[2]['names'] = [name.replace('_', '\_') for name in plots_dict_list[2]['names']]
      return plots_dict_list

  @staticmethod
  def wandb_read_fn(csv_path_list):
      plots_dict_list = []
      for i in range(len(csv_path_list)):
          plots_dict = {}
          
          path = csv_path_list[i]
          raw  = pandas.read_csv(path, header=0)
          df   = pandas.DataFrame(raw)
          min_name = sorted([name for name in list(df.columns)[1:] if "__MIN" in name])
          max_name = sorted([name for name in list(df.columns)[1:] if "__MAX" in name])
          plots_dict['names']       = sorted([name for name in list(df.columns)[1:] if "__M" not in name]) 
          plots_dict['df']          = [pandas.DataFrame(df[name]) for name in plots_dict['names']]
          plots_dict['std']         = []
          idx = 0
          for name in plots_dict['names']:
              if name in min_name[idx]:
                  plots_dict['std'].append({"MIN": pandas.DataFrame(df[min_name[idx]]), "MAX":pandas.DataFrame(df[max_name[idx]])})
                  idx += 1
              else:
                  plots_dict['std'].append(None)
          for d in plots_dict['df']: d.columns = ['y']
          plots_dict['xlabel']      = "Communication rounds"
          plots_dict['ylabel']      = r"Accuracy($\%$) $\uparrow$"
          plots_dict['y_lim']       = None 
          plots_dict['yticks']      = None 
          plots_dict['xticks']      = None
          plots_dict['yticklabels'] = None
          plots_dict['xticklabels'] = None
          plots_dict['alpha']       = 0.15
          plots_dict['smooth']      = False
          plots_dict['sigma']       = True if len(min_name) > 0 else False
          plots_dict['markon']      = True
          
          plots_dict_list.append(plots_dict) 
      #=== special settings for each plots === 
      # plots_dict_list[1]['names'] = [name.replace('_', '\_') for name in plots_dict_list[1]['names']]
      # plots_dict_list[2]['names'] = [name.replace('_', '\_') for name in plots_dict_list[2]['names']]

      for idx, name in enumerate(plots_dict['names']):
          if 'bn' in name:
              plots_dict['names'][idx] = "BatchNorm"
          elif 'ln' in name:
              plots_dict['names'][idx] = "LayerNorm"
          elif 'gn' in name:
              plots_dict['names'][idx] = "GroupNorm"
          elif 'no' in name:
              plots_dict['names'][idx] = "NoNorm"   
          elif 'fbn' in name:
              plots_dict['names'][idx] = "Fixed BatchNorm"
          elif 'in' in name:
              plots_dict['names'][idx] = "InstanceNorm" 

      return plots_dict_list
  

class Plotter:
  ###========== Plot publication quality figures ============ ###
  @staticmethod
  def plot_figure(csv_path_list,
                  plot_fn,
                  read_fn,
                  num_axis=(1, 1), 
                  legenddict = {'bbox':(0.253, 1.237), 'ncol':1},
                  pad = 0.0,
                  width = 3.487,
                  height = 3.487 / 1.618,
                  text_list = [],
                  save_path = './plot.pdf',
                  bbox_inches = 'tight',
                  ):
      
      #============ main part of the plot figure code ===============
      # matplotlib.style.use('seaborn')
    
      plt.rc('font', family='serif', serif='Times')
      plt.rc('text', usetex=True)
      plt.rc('xtick', labelsize=12)
      plt.rc('ytick', labelsize=12)
      plt.rc('axes', labelsize=12)
      
      fig, ax = plt.subplots(num_axis[0], num_axis[1])
      plt.show(block=False)
      
      if min(num_axis[0], num_axis[1]) > 1:
          ax = [ax[i,j] for i in range(num_axis[0]) for j in range(num_axis[1])]

      plots_dict_list = read_fn(csv_path_list)
      if num_axis[0]*num_axis[1] > 1:
        for i in range(len(ax)):
            plot_fn(ax = ax[i], plots_dict = plots_dict_list[i])
      else:
        plot_fn(ax = ax, plots_dict = plots_dict_list[0])
      
      #=== global legends ===
      # handles, labels = ax[0].get_legend_handles_labels()
      # fig.legend(handles, labels, bbox_to_anchor=legenddict['bbox'], ncol=legenddict['ncol'], loc='center', fontsize=16)    
      # fig.tight_layout(pad=pad)
          
      for text in text_list:
        plt.text(text['x'], text['y'], text['text'], fontsize=text['fontsize'])

      #fig.set_size_inches(width, height)
      fig.tight_layout()
      fig.savefig(save_path, bbox_inches=bbox_inches)

  @staticmethod
  def normal_plot_fn(ax, plots_dict):
    df_list     = plots_dict['df']           #list
    full_names  = plots_dict['names']        #list
    xlabel      = plots_dict['xlabel']       #str
    ylabel      = plots_dict['ylabel']       #str
    y_lim       = plots_dict['y_lim']        #list
    yticks      = plots_dict['yticks']       #list
    xticks      = plots_dict['xticks']       #list
    yticklabels = plots_dict['yticklabels']  #list
    xticklabels = plots_dict['xticklabels']  #list
    alpha       = plots_dict['alpha']
    
    c = ['darkblue', 'darkgreen', 'darkorange', 'darkred', 'darkslategray', 'darkmagenta', 'gold']
    m = ['s', 'o', '^', '*', "X", 'D']
    
    for i in range(len(df_list)):
      y = df_list[i]['y'].to_numpy()[1:]
      x = np.arange(len(y))
      markon = x if len(x) < 10 else x[::len(x)//10]
      if plots_dict['smooth']:
        x_smooth = np.linspace(x.min(), x.max(), 100) 
        spl = make_interp_spline(x, y, k=2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, label=full_names[i], color=c[i], marker=m[i] if plots_dict['markon'] else None, markevery=markon)
      else:
        ax.plot(x, y, marker=m[i] if plots_dict['markon'] else None, label=full_names[i], color=c[i], markevery=markon)
      
      if plots_dict['sigma']:
        mu    = df_list[i]['mean'].to_numpy()
        sigma = df_list[i]['std'].to_numpy()
        if i == 2:   
          ax.fill_between(x, mu+sigma, mu-sigma, facecolor=c[i], alpha=alpha*2)
        else:
          ax.fill_between(x, mu+sigma, mu-sigma, facecolor=c[i], alpha=alpha)
  
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if xlabel      is not None: ax.set_xlabel(xlabel, size=12)
    if ylabel      is not None: ax.set_ylabel(ylabel,size=12)
    if y_lim       is not None: ax.set_ylim(y_lim)
    if xticks      is not None: ax.set_xticks(xticks)
    if yticks      is not None: ax.set_yticks(yticks)
    if xticklabels is not None: ax.set_xticklabels(xticklabels)
    if yticklabels is not None: ax.set_yticklabels(yticklabels)
    ax.legend(loc='upper left')
    ax.grid(True)


  @staticmethod
  def wandb_plot_fn(ax, plots_dict):
    df_list     = plots_dict['df']           #list
    full_names  = plots_dict['names']        #list
    xlabel      = plots_dict['xlabel']       #str
    ylabel      = plots_dict['ylabel']       #str
    y_lim       = plots_dict['y_lim']        #list
    yticks      = plots_dict['yticks']       #list
    xticks      = plots_dict['xticks']       #list
    yticklabels = plots_dict['yticklabels']  #list
    xticklabels = plots_dict['xticklabels']  #list
    alpha       = plots_dict['alpha']
    std         = plots_dict['std']          #list
    
    c = ['darkblue', 'darkgreen', 'darkorange', 'darkred', 'darkslategray', 'darkmagenta', 'gold']
    m = ['s', 'o', '^', '*', "X", 'D']
    
    for i in range(len(df_list)):
      y = np.asarray(df_list[i]['y'][:-1].to_numpy(), dtype=np.float32)
      x = np.arange(len(y))
      y = y[::100]
      x = x[::100]
      markon = len(x)//10
      if plots_dict['smooth']:
        x_smooth = np.linspace(x.min(), x.max(), 100) 
        spl = make_interp_spline(x, y, k=2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, label=full_names[i], color=c[i], marker=m[i] if plots_dict['markon'] else None, markevery=markon)
      else:
        ax.plot(x, y, marker=m[i] if plots_dict['markon'] else None, label=full_names[i], color=c[i], markevery=markon)
      
      if plots_dict['sigma'] and std[i] is not None:
        min_std = std[i]['MIN'][:-1][::100]
        min_std.interpolate(method='nearest')
        min_std = min_std.to_numpy().reshape(-1)

        max_std = std[i]['MAX'][:-1][::100]
        max_std.interpolate(method='nearest')
        max_std = max_std.to_numpy().reshape(-1)
        if i == 2:   
          ax.fill_between(x, min_std, max_std, facecolor=c[i], alpha=alpha*2)
        else:
          ax.fill_between(x, min_std, max_std, facecolor=c[i], alpha=alpha)
  
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if xlabel      is not None: ax.set_xlabel(xlabel, size=18)
    if ylabel      is not None: ax.set_ylabel(ylabel, size=18)
    if y_lim       is not None: ax.set_ylim(y_lim)
    if xticks      is not None: ax.set_xticks(xticks)
    if yticks      is not None: ax.set_yticks(yticks)
    if xticklabels is not None: ax.set_xticklabels(xticklabels)
    if yticklabels is not None: ax.set_yticklabels(yticklabels)
    ax.legend(loc='lower right', ncol=2, fontsize=16)
    ax.grid(True)


if __name__ == "__main__":
    reader  = Reader()
    plotter = Plotter()

    plotter.plot_figure(["../../data/plots/resnet.csv"], plotter.wandb_plot_fn, reader.wandb_read_fn, save_path='resnet.pdf')
    # print(plots_dict_list[0]['df'])
    print(plots_dict_list[0]['names'])