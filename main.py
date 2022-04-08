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
s /= s.sum() # normalized PSF
im = ift2(fft2(ob)*fft2(s))

#This is where we do our deconvolution. This algorithm is called Richardson-Lucy
#deconvolution, and is very popular.
ni = 100
decon = rl_basic(im, s, n_iter = ni)

#After deconvolution, we apply the bandlimit to the deconvolved object, so
#we can compare it to the true bandlimit object we started with, and check
#how accurate our deconvolution was.
decon_bl = ift2(fft2(decon)*S_bl)

print(f"\nRL MSE: {np.sqrt((decon_bl-ob_bl)**2).mean():.2e}")
#imshow(ob_bl)
#imshow(decon_bl)



# compute KL divergence
def KL(f, g, K_ft, b):
    Af = ift2(fft2(f)*K_ft)
    temp = g * np.log( g/(Af + b) ) + Af + b - g
    return np.sum(temp)

# compute gradient of KL divergence
def gradKL(f, g, K, K_ft, K_flip_ft, b):
    Af = ift2(fft2(f)*K_ft)
    quotient = g/(Af + b)
    ATquotient = ift2( fft2(quotient) * K_flip_ft )
    return np.ones_like(ATquotient) - ATquotient

# algorithm input
f = im.copy()
g = im.copy()
K = s
K_ft = fft2(K)
K_flip = np.roll(np.flip(s), 1, axis=(0,1))
K_flip_ft = fft2(K_flip)
b =  np.zeros_like(f)
beta = 1e-4
theta = 0.4

# optimization parameters
alpha = 1.3 # alpha_0
tau = 0.5   # tau_1
alpha_max = 1e5
alpha_min = 1e-5

# one step of RL to optimize L_1 and L_2
Af = ift2(fft2(f)*K_ft)
quotient = g/(Af + b)
ATquotient = ift2( fft2(quotient) * K_flip_ft )
y = f*ATquotient
y_max = y.max()
y_min = y.min()

if y_max/y_min < 50:
    L_1 = y_min/10
    L_2 = y_max/10
else:
    L_1 = y_min
    L_2 = y_max

L_1_arr = L_1 + np.zeros_like(f)
L_2_arr = L_2 + np.zeros_like(f)

print("rl kl-divergence = {}".format( KL(decon, g, K_ft, b) ))
print("initial kl-divergence = {}".format( KL(f,g,K_ft,b) ))

# value initialization
a_2_prev2 = np.inf
a_2_prev  = np.inf
a_2       = np.inf
gradient = gradKL(f, g, K, K_ft, K_flip_ft, b)

# SGP algorithm
for k in range(25):
    # Choose alpha_k and D_k
    D_diag = np.minimum(L_2_arr, np.maximum(L_1_arr, f))

    if k > 0:
        alpha_prev = alpha
        gradient_prev = gradient
        gradient = gradKL(f, g, K, K_ft, K_flip_ft, b)
        temp_z = gradient - gradient_prev

        denominator_BB1 = np.vdot(temp_s, temp_z/D_diag)
        numerator_BB2 = np.vdot(temp_s, D_diag*temp_z)

        a_2_prev2 = a_2_prev
        a_2_prev = a_2

        if denominator_BB1 <= 0:
            a_1 = min(10*alpha_prev, alpha_max)
        else:
            alpha_BB1 = np.vdot(temp_s, temp_s/(D_diag*D_diag)) / denominator_BB1
            a_1 = min(alpha_max, max(alpha_min, alpha_BB1))
        
        if numerator_BB2 <= 0:
            a_2 = min(10*alpha_prev, alpha_max)
        else:
            alpha_BB2 = numerator_BB2 / np.vdot(temp_z, D_diag*D_diag*temp_z)
            a_2 = min(alpha_max, max(alpha_min, alpha_BB2))
        
        if a_2/a_1 <= tau:
            alpha = min(a_2_prev2, a_2_prev, a_2)
            tau *= 0.9
        else:
            alpha = a_1
            tau *= 1.1
        

    # projection
    y = f - alpha*D_diag*gradient
    y[y < 0] = 0

    # decent direction
    d = y-f

    # backtracking loop
    c = 1
    KLdivergence = KL(f,g,K_ft,b)
    while True:
        J_new = KL(f + c*d, g, K_ft, b)
        if J_new <= KLdivergence + beta*c*np.vdot(gradient, d):
            break
        else:
            c *= theta
    
    temp_s = c*d
    f += temp_s

    #print("k={}    div={}".format(k, KL(f,g,K_ft,b)))
    #f_bl = ift2(fft2(f)*S_bl)
    #print(f"SGP MSE: {np.sqrt((f_bl-ob_bl)**2).mean():.2e}")

print("final kl-divergence = {}".format( KL(f,g,K_ft,b) ))
f_bl = ift2(fft2(f)*S_bl)
print(f"SGP MSE: {np.sqrt((f_bl-ob_bl)**2).mean():.2e}")

# figure out influence of norm on mean squared error
print("")
print("norm of bl SGP = {}".format( np.linalg.norm(f_bl) ))
print("norm of bl RL = {}".format( np.linalg.norm(decon_bl) ))
print("norm of bl object = {}".format( np.linalg.norm(ob_bl) ))

decon_bl_scaled = decon_bl * ( np.linalg.norm(ob_bl) / np.linalg.norm(decon_bl) )
f_bl_scaled = f_bl * ( np.linalg.norm(ob_bl) / np.linalg.norm(f_bl) )

print(f"Scaled RL MSE: {np.sqrt((decon_bl_scaled-ob_bl)**2).mean():.2e}")
print(f"Scaled SGP MSE: {np.sqrt((f_bl_scaled-ob_bl)**2).mean():.2e}")
print("Scaled bl RL KL-div: {}".format(KL(decon_bl_scaled,g,K_ft,b)))
print("Scaled bl SGP KL-div: {}".format(KL(f_bl_scaled,g,K_ft,b)))

print("")


# show and compare results
print("loop complete. showing result")
imshow(f)
print("showing bl result")
imshow(f_bl)
# print("showing rl algorithm")
# imshow(decon)
print("showing rl algorithm bl")
imshow(decon_bl)
# print("showing original object")
# imshow(ob)
print("showing bl object")
imshow(ob_bl)
# print("showing convolved object")
# imshow(im)
