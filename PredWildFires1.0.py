# -*- coding: utf-8 -*-
"""
Title: PredWildFires.py
Date: Fall 2019
Description:  Calls a Gui, reads data files, builds model, and then runs
data against the model.
"""
from mpl_toolkits.basemap import Basemap  #For the mapping package
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wx
import os
#Create the App Structure

class CanvasFrame(wx.Frame):
   def __init__(self): 
       app = wx.App(False)
       wx.Frame.__init__(self,None,-1,'Wildfire Predictions',size=(800,600))
       self.figure = Figure(figsize = (12,8), dpi=80 )
       self.ax = self.figure.add_subplot(121)
       self.ax2 = self.figure.add_subplot(122) 
       self.canvas = FigureCanvas(self, -1, self.figure)
       self.canvas2 = FigureCanvas(self, -1, self.figure)
       self.sizer = wx.BoxSizer(wx.VERTICAL)
       self.sizer2 = wx.BoxSizer(wx.VERTICAL)
       self.sizer.Add(self.canvas, 1, wx.ALIGN_LEFT | wx.ALIGN_TOP)
       self.sizer2.Add(self.canvas2, 1, wx.ALIGN_RIGHT | wx.ALIGN_TOP)
       self.SetSizer(self.sizer)
       self.SetSizer(self.sizer2)
       self.Fit()
       self.map = Basemap(projection = 'mill',
                     llcrnrlat = 36,
                     llcrnrlon = -125,
                     urcrnrlat = 43,
                     urcrnrlon = -116,
                     resolution = 'h',
                     ax=self.ax)
       self.map.drawcoastlines()
       self.map.drawcountries()
       self.map.drawstates()
       self.map.bluemarble()
       self.map.drawrivers(linewidth=0.5, linestyle='solid', color='blue')
       self.figure.canvas.draw()
       self.Show(True)
       app.MainLoop()
       

   def OnExit(self,e):
        self.Close(True)  # Close the frame.


frame = CanvasFrame()    


