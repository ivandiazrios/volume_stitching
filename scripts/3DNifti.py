#!/usr/bin/python

from __future__ import print_function
import numpy
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys

class IndexTracker(object):
    def __init__(self, ax, X, axis):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        rows, cols, depths = X.shape

        if axis == 'x':
            self.func = lambda: self.X[self.ind,:,:]
            self.slices = rows
        elif axis == 'y':
            self.func = lambda: self.X[:,self.ind,:]
            self.slices = cols
        elif axis == 'z':
            self.func = lambda: self.X[:,:,self.ind]
            self.slices = depths
        else:
            raise ValueError("Axis %s needs to be x, y or z" % axis)

        self.X = X

        self.ind = self.slices/2

        self.im = ax.imshow(self.func())
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        print(self.ind)
        if event.button == 'up':
            self.ind = numpy.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = numpy.clip(self.ind - 1, 0, self.slices - 1)

        self.update()

    def update(self):
        self.im.set_data(self.func())
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

fig = plt.figure()
ax = fig.add_subplot(111)

plt.set_cmap("gray")

args_len = len(sys.argv)

if len(sys.argv) not in [2,3]:
    print("Command line args: path to file, (optional) axis letter (X,Y,Z)")

image = sitk.ReadImage(sys.argv[1])
X = sitk.GetArrayFromImage(image)

if args_len == 3:
    axis = str.lower(sys.argv[2])
else:
    axis = 'x'

tracker = IndexTracker(ax, X, axis)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()