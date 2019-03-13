# -*- coding: utf-8 -*-
"""
Demonstrate a simple data-slicing task: given 3D data (displayed at top), select 
a 2D plane and interpolate data along that plane to generate a slice image 
(displayed at bottom). 


"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800,800)
win.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)
imv1 = pg.ImageView()
imv2 = pg.ImageView()
l.addWidget(imv1, 0, 0)
l.addWidget(imv2, 1, 0)

# Create jet colormap
jet_rgba = np.load('jet.npy')
jet_vals = np.linspace(0, 1, 256)
jet_cmap = pg.ColorMap(jet_vals, jet_rgba)
jet_lut = jet_cmap.getLookupTable(0.0, 1.0, 256)
imv1.ui.histogram.gradient.setColorMap(jet_cmap)
imv2.ui.histogram.gradient.setColorMap(jet_cmap)
# imv1.getImageItem().setLookupTable(jet_lut)
# imv2.getImageItem().setLookupTable(jet_lut)

win.show()

roi = pg.LineSegmentROI([[10, 10], [80,80]], pen='r')
imv1.addItem(roi)

data = np.load('data_grid.npy')

def update():
    global data, imv1, imv2
    d2 = roi.getArrayRegion(data, imv1.imageItem, axes=(0,2))
    imv2.setImage(d2[:, ::-1])
    
roi.sigRegionChanged.connect(update)

## Display the data
imv1.setImage(data[:, 345//2, :])
# imv1.setHistogramRange(-0.01, 0.01)
# imv1.setLevels(-0.003, 0.003)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
