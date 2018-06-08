'''
Created on 2 Jan 2015

@author: jadarve
'''


import pkg_resources
import json

from multiprocessing import Pool
from threading import Semaphore
import signal


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

from . import flowplot


__all__ = ['createPlot', 'getLayout', 'plot',
           'plotFlow', 'plotImage', 'plotScalarField', 'plot1D', 'plotMulti1D']


def createPlot(rows=1, cols=1, height=5, width=5, tight=True, suptitle=None):
    """
    Creates a new plot descriptor

    Parameters
    ----------
    rows: number of rows
    cols: number of columns
    height: height in centimeters
    width: width in centimeters
    tight: whether or not the plot layout is tight
    suptitle : string. Figure super title in the top part of the plot. Defaults to None.

    Returns
    -------
    map structure with the descriptor of the plot
    """
    try:
        fp = open(
            pkg_resources.resource_filename('pltutil.rsc', 'plot.json'))
        desc = json.load(fp)
        desc['width'] = width
        desc['height'] = height
        desc['rows'] = rows
        desc['columns'] = cols
        desc['tight'] = tight
        desc['suptitle'] = suptitle
        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')


def addImageAxis(pos, span=(1, 1), data=None, title=None, cmap=None):

    try:
        fp = open(pkg_resources.resource_filename(
            'pltutil.rsc', 'image.json'))
        desc = json.load(fp)
        desc['pos'] = pos
        desc['span'] = span
        desc['data'] = data
        desc['title'] = title
        desc['cmap'] = cmap
        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')


def addScalarFieldAxis(pos, span=(1, 1), data=None, title=None, vmin=None, vmax=None, axis=True):

    try:
        fp = open(pkg_resources.resource_filename(
            'pltutil.rsc', 'scalarfield.json'))
        desc = json.load(fp)
        desc['pos'] = pos
        desc['span'] = span
        desc['data'] = data
        desc['title'] = title
        desc['min'] = vmin
        desc['max'] = vmax
        desc['axis'] = axis
        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')


def addFlowAxis(pos, span=(1, 1), data=None, maxflow=1.0, title=None):

    try:
        fp = open(
            pkg_resources.resource_filename('pltutil.rsc', 'flow.json'))
        desc = json.load(fp)
        desc['pos'] = pos
        desc['span'] = span
        desc['data'] = data
        desc['title'] = title
        desc['max'] = maxflow
        desc['color_wheel'] = True
        desc['color_wheel_size'] = 150
        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')


def addColorWheelAxis(pos, span=(1,1), maxflow=1.0, ticks=True):
    try:
        fp = open(pkg_resources.resource_filename('pltutil.rsc', 'colorwheel.json'))
        desc = json.load(fp)
        desc['pos'] = pos
        desc['span'] = span
        desc['maxflow'] = maxflow
        desc['ticks'] = ticks
        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')
    

def addPlot1DAxis(pos, span=(1, 1), Xdata=None, Ydata=None, title=None):
    try:
        fp = open(pkg_resources.resource_filename('pltutil.rsc', '1D.json'))
        desc = json.load(fp)
        desc['pos'] = pos
        desc['span'] = span
        desc['Y'] = Ydata
        desc['X'] = Xdata
        desc['title'] = title
        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')


def addMulti1DAxis(pos, span=(1, 1), Xdata=None, series=None, title=None):
    try:
        fp = open(pkg_resources.resource_filename(
            'pltutil.rsc', 'multi1D.json'))
        desc = json.load(fp)
        desc['pos'] = pos
        desc['span'] = span
        desc['title'] = title
        desc['X'] = Xdata

        if series != None:
            # a series element is composed by (label, data, color, line_width)
            for s in series:
                desc['series'][s[0]] = dict()
                desc['series'][s[0]]['label'] = s[0]
                desc['series'][s[0]]['data'] = s[1]
                desc['series'][s[0]]['color'] = s[2]
                desc['series'][s[0]]['line_width'] = s[3]

        return desc
    except IOError:
        raise IOError('error reading json plot file')
    except ValueError:
        raise ValueError('malformed JSON text')


def getLayout(name):

    fileName = pkg_resources.resource_filename(
        'pltutil.rsc', '{0:s}.json'.format(name))
    try:
        fp = open(fileName)
        return json.load(fp)
    except IOError:
        raise IOError('error reading plot layout: {0:s}'.format(name))
    except ValueError:
        raise ValueError('malformed JSON text for layout: {0:s}'.format(name))


def plot(desc):

    global __plotMethods__

    width = float(desc['width']) / 2.54
    height = float(desc['height']) / 2.54
    rows = int(desc['rows'])
    cols = int(desc['columns'])

    # creates the figure
    fig = plt.figure(figsize=(width, height))
    if desc['tight']:
        fig.set_tight_layout(True)
    
    if desc['suptitle'] != None:
        fig.suptitle(desc['suptitle'])

    # creates the figure layout
    layout = Layout(fig, (rows, cols))

    try:
        # for each plot
        plots = desc['plots']
        for plotName in plots.keys():
    
            plotDesc = plots[plotName]
            pos = plotDesc['pos']
            span = plotDesc['span']
    
    #         print(plotName)
    #         print(pos)
    #         print(span)
    
            # creates an axis for the plot
            axis = layout.createAxis(pos, span)
    
            # call the appropriate plotting method according
            # the the class attribute
            plotMethod = __plotMethods__[plotDesc['class']]
            plotMethod(axis, plotDesc)
            
    except Exception as e:
        print(e)

    # returns the figure
    return fig


def plotFlow(axis, desc):

    flowColor = flowplot.flowToColor(desc['data'], desc['max'])
    
    dshape = desc['data'].shape
    
    axis.set_xlim(0, dshape[1])
    axis.set_ylim(dshape[0], 0)
    
    axis.imshow(flowColor, extent=(0, dshape[1], dshape[0], 0))

    if 'title' in desc.keys() and desc['title'] != None:
        axis.set_title(desc['title'])

    if 'axis' in desc.keys():
        if not desc['axis']:
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())
    
    
    # color wheel
    if 'color_wheel' in desc.keys() and desc['color_wheel']:
        cwheelSize = desc['color_wheel_size']
        axis.imshow(flowplot.colorWheel(), extent=(dshape[1] - cwheelSize, dshape[1], 
                                                   dshape[0], dshape[0] - cwheelSize))
    
    
    if 'divider' in desc.keys():
        
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size=desc['divider'], pad=0.05)
        cax.axis('off')
#     plt.colorbar(im, cax=cax, ticks=plticker.MaxNLocator(nbins))


    if 'x_label' in desc.keys():
        axis.set_xlabel(desc['x_label'])
        
    if 'y_label' in desc.keys():
        axis.set_ylabel(desc['y_label'])
        

def plotColorWheel(axis, desc):
    
    maxflow = desc['maxflow']
    
    axis.imshow(flowplot.colorWheel(), extent=[-maxflow, maxflow, -maxflow, maxflow])
    
    # location of the coordinate frame spines
    axis.spines['left'].set_position('center')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_position('center')
    axis.spines['top'].set_color('none')
    axis.spines['left'].set_smart_bounds(True)
    axis.spines['bottom'].set_smart_bounds(True)
    
    # tick labels
    ticks_pos = np.arange(-maxflow, maxflow, 1)
    tick_labels = range(-int(maxflow), int(maxflow), 1)
    tick_labels[0] = ''
    tick_labels[len(tick_labels)/2] = ''
    
    if 'ticks' in desc.keys():
        if not desc['ticks']:
            tick_labels[:] = ''

    axis.xaxis.set_ticks(ticks_pos)
    axis.xaxis.set_ticklabels(tick_labels)
    axis.xaxis.set_ticks_position('bottom')

    axis.yaxis.set_ticks(ticks_pos)
    axis.yaxis.set_ticklabels(tick_labels)
    axis.yaxis.set_ticks_position('left')
    

def plotImage(axis, desc):

    axis.imshow(desc['data'], cmap=plt.cm.get_cmap(desc['cmap']))

    if 'title' in desc.keys() and desc['title'] != None:
        axis.set_title(desc['title'])

    if 'axis' in desc.keys():
        if not desc['axis']:
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())
            
    
    if 'x_label' in desc.keys():
        axis.set_xlabel(desc['x_label'])
        
    if 'y_label' in desc.keys():
        axis.set_ylabel(desc['y_label'])
        
    if 'divider' in desc.keys():
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size=desc['divider'], pad=0.05)
        cax.axis('off')


def plotScalarField(axis, desc):

    cmap = desc['cmap'] if 'cmap' in desc.keys() else 'gray'
    vmin = desc['min'] if 'min' in desc.keys() else None
    vmax = desc['max'] if 'max' in desc.keys() else None

    im = axis.imshow(desc['data'], vmin=vmin, vmax=vmax,
                     cmap=plt.cm.get_cmap(cmap))

    if 'title' in desc.keys() and desc['title'] != None:
        axis.set_title(desc['title'])

    if 'axis' in desc.keys():
        if not desc['axis']:
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())
            
    if 'x_label' in desc.keys():
        axis.set_xlabel(desc['x_label'])
        
    if 'y_label' in desc.keys():
        axis.set_ylabel(desc['y_label'])

    if 'cbar' in desc.keys() and desc['cbar']:
        nbins = 4 if 'cbar_bins' not in desc.keys() else  desc['cbar_bins']
        locs = None if 'cbar_locs' not in desc.keys() else  desc['cbar_locs']
        size = '3%' if 'cbar_size' not in desc.keys() else desc['cbar_size']
        labels = None if 'cbar_labels' not in desc.keys() else desc['cbar_labels']
        addColorbar(axis, im, size=size, locs=locs, nbins=nbins, labels=labels)
    
    if 'divider' in desc.keys():
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size=desc['divider'], pad=0.05)
        cax.axis('off')
        

def plot1D(axis, desc):

    Ydata = desc['Y']

    Xdata = desc['X'] if 'X' in desc.keys() and desc['X'] != None \
        else np.arange(Ydata.shape[0])

    lw = desc['line_width'] if 'line_width' in desc.keys() else 1

    axis.plot(Xdata, Ydata, linewidth=lw)
    axis.set_xlim([0, Xdata.shape[0]])

    if 'title' in desc.keys():
        axis.set_title(desc['title'])

    if 'axis' in desc.keys():
        if not desc['axis']:
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())

    if 'marker' in desc.keys():
        if desc['marker']:
            i = desc['marker_index']
            axis.scatter([Xdata[i]], [Ydata[i]],
                         c=desc['marker_color'], s=desc['marker_size'])


def plotMulti1D(axis, desc):

    Xdata = desc['X']

    series = desc['series']
    handles = []
    for k in series.keys():
        s = series[k]
        h, = axis.plot(Xdata, s['data'], linewidth=s['line_width'], color=s[
                  'color'], label=s['label'], zorder=0)
        handles.append(h)

    axis.set_xlim(0, Xdata[-1])
    axis.set_ylim(desc['y_min'] if 'y_min' in desc.keys() else None,
                  desc['y_max'] if 'y_max' in desc.keys() else None)
    
    plt.legend(handles=handles, loc=1)

    if 'x_label' in desc.keys() and desc['x_label'] != None:
        axis.set_xlabel(desc['x_label'])

    if 'y_label' in desc.keys() and desc['y_label'] != None:
        axis.set_ylabel(desc['y_label'])

    if 'title' in desc.keys() and desc['title'] != None:
        axis.set_title(desc['title'])

    if 'axis' in desc.keys():
        if not desc['axis']:
            axis.xaxis.set_major_locator(plt.NullLocator())
            axis.yaxis.set_major_locator(plt.NullLocator())

    if 'marker' in desc.keys() and desc['marker']:
        i = desc['marker_index']

        markerX = np.zeros((len(series)))
        markerY = np.zeros((len(series)))
        colors = list()

        c = 0
        for k in series.keys():
            s = series[k]
            markerX[c] = Xdata[i]
            markerY[c] = s['data'][i]
            colors.append(s['color'])
            c += 1

        axis.scatter(
            markerX, markerY, c=colors, s=desc['marker_size'], zorder=1)


def addColorbar(ax, im, size='3%', locs=None, nbins=4, labels=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=0.05)
    
    if locs == None:
        cbar = plt.colorbar(im, cax=cax, ticks=plticker.MaxNLocator(nbins))
            
    else:
        cbar = plt.colorbar(im, cax=cax, ticks=plticker.FixedLocator(locs, nbins))
        
        if labels is not None:
            cbar.ax.set_yticklabels(labels)


# look-up table for the different plot methods supported
__plotMethods__ = {'FLOW': plotFlow,
                   'IMAGE': plotImage,
                   'SCALAR': plotScalarField,
                   '1D': plot1D,
                   'MULTI1D': plotMulti1D,
                   'COLOR_WHEEL': plotColorWheel}


class Layout(object):

    ###################################
    # INSTANCE ATTRIBUTES
    ###################################

    parent = None

    childs = None

    # global offset
    loc = (0, 0)

    # local size
    size = (0, 0)

    # global size
    gsize = (0, 0)

    # figure container
    fig = None

    ###################################
    # CONSTRUCTORS
    ###################################

    def __init__(self, fig, size=(1, 1), loc=(0, 0), parent=None):

        if fig is None:
            raise Exception('fig parameter None')

        self.fig = fig
        self.childs = list()
        self.size = size
        self.loc = loc

        if parent is None:
            self.gsize = size
        else:
            self.gsize = parent.gsize

    ###################################
    # PUBLIC METHODS
    ###################################

    def createChild(self, loc, size=(1, 1)):
        """
        Creates a child layout with a given location *loc* and *size*
        """

        cloc = (self.loc[0] + loc[0], self.loc[1] + loc[1])
        child = Layout(self.fig, size, cloc, self)
        self.childs.append(child)
        return child

    def createAxis(self, loc, size=(1, 1), **kargs):
        """
        Creates a 2D axis in local coordinates
        """

        axloc = (self.loc[0] + loc[0], self.loc[1] + loc[1])

        subplotspec = GridSpec(self.gsize[0],
                               self.gsize[1]).new_subplotspec(axloc,
                                                              rowspan=size[0],
                                                              colspan=size[1])

        # TODO: check that gLoc + size does not go outside the boundaries of
        # this layout
        ax = self.fig.add_subplot(subplotspec, **kargs)
        return ax

    ###################################
    # PRIVATE METHODS
    ###################################


