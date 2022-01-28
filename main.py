#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:42:47 2020

@author: nikolaj
"""

import numpy as np
from src.zern import get_zern, normalize
from src.image_functions import scl, sft, ift, fft, fft2, ift2, imshow
from src.sim_cell import cell_multi
from deconvolution.deconv import rl_basic


#precompute pupil and zernike polynomials
pixelSize = .096
NA = 1.2
l = .532
RI = 1.33
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2
l2p = 2*np.pi/l #length to phase

dsize = 128
dim = (dsize, dsize)
num_phi = 42
rotang = 0

zern, R, Theta, inds, inds2 = get_zern(dsize, pupilSize, pixelSize, num_phi, p2 = True)
#dont worry about any of the above code yet

ob = cell_multi(dsize, 100, (20, 22), e = .1, overlap = .5)
ob = scl(ob)


#First we want to create a PSF that gives us our bandlimited object

#In deconvolution the goal is to recover the object, but due to the physics of
#how microscopes work, only a part of the object can be recovered, up to the
#bandlimit, which is the highest frequency that can be transmitted by the 
#microscope.

#In the frequency domain, this equates to an array of ones and zeros, which
#specify the freqs that make it through the microscope and those that dont. 
#Feel free to plot stuff to look at things like the PSF, frequency representation
#of the PSF (also called the OTF)

S_bl = np.ones(dim)
S_bl[inds2] = 0
S_bl = sft(S_bl)[:,:dsize//2+1]

#This is how we convolve stuff, by taking the fft of the two things we want to
#convolve, multiplying them, and then taking the inverse fft of that product. I
#use a special kind of fft called a Hermitian fft, which just assumes that the 
#input is always real, and exploits that for some performance improvements. 
ob_bl = ift2(fft2(ob)*S_bl)
#note that the object above can not be deconvolved- thats what the machine
#learning part of this project will be about. The only reason I'm including this
#object is that this is what the deconvolved result will be compared against,
#not the pristine object (ob).

#Here is where we define the PSF of an ideal microscope, and convolve it with
#the object to simulate what we'd see through that ideal microscope
H = np.zeros(dim)
H[inds] = 1
h = ift(H)
s = np.abs(h)**2
im = ift2(fft2(ob)*fft2(s))

#This is where we do our deconvolution. This algorithm is called Richardson-Lucy
#deconvolution, and is very popular.
ni = 100
decon = rl_basic(im, s, n_iter = ni)

#After deconvolution, we apply the bandlimit to the deconvolved object, so
#we can compare it to the true bandlimit object we started with, and check
#how accurate our deconvolution was.
decon_bl = ift2(fft2(decon)*S_bl)

print(f"\nMSE: {np.sqrt((decon_bl-ob_bl)**2).mean():.2e}")
# imshow(ob_bl)
# imshow(decon_bl)


