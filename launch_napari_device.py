#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 21:12:58 2022

@author: debian
"""
import tensorflow
import napari
with tensorflow.device('/cpu:0'):
     print('launching')
     napari.Viewer()
     napari.run()