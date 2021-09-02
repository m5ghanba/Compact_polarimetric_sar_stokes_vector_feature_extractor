# Deriving features from CP 2*2 coherence matrix (or equivalently Stokes vector),   Mohsen Ghanbari May 4, 2020

# Input: the elements of coherence matrix: c11, c12_real, c22, c12_imag
# Output: The derived CP features:
# 1) scattering mechanism, alpha_s     2) circular polarization ratio, miu_c            3) conformity coefficient (ellipticity), u
# 4) correlation coefficient, rho      5) relative phase angle between RH and Rv, delta 6) degree of polarizatio, m
# 7) Shannon entropy, intensity, h_i   8) Shannon entropy, polarimetric component, h_p  9) degree of linear polar, m_l
#10) linear polarization ratio, miu_l 11) orientation of the ellipse, psi              12) axial ratio of the ellipse, r
#13 - 15) m-chi decomp, mchi_b, mchi_r, mchi_g              16 - 18) m-delta decomp, m_delta_r, m_delta_g, m_delta_b
#19 - 20) RH and RV intensities, rh, rv                     21 - 24) Stokes parameters, s0, s1, s2, s3
import scipy.io
import numpy as np
import math
from PIL import Image
import os
from enum import Enum
import tifffile as tiff
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Use the script CP_Preprocessing_Downsampling_Multilookaveraging.py to generate coh elements.
#p_directory = 'C:/Users/vip-mohsen/Desktop/Results_realCp/Montreal/Montreal 1/box5/'
p_directory = askdirectory(title='Select Folder') # shows dialog box and return the path
p_directory = p_directory + "/"

matfile = scipy.io.loadmat(p_directory + 'CoherenceMatrixElements.mat')  # c11, c12_real, c22, c12_imag
c11 = matfile['c11']
c12_real = matfile['c12_real']
c12_imag = matfile['c12_imag']
c22 = matfile['c22']



# stokes = tiff.imread(p_directory + 'SV.tif')
# s0 = stokes[:,:,0]
# s1 = stokes[:,:,1]
# s2 = stokes[:,:,2]
# s3 = stokes[:,:,3]
#
# c11 = 0.5 * (s0 + s1)
# c22 = 0.5 * (s0 - s1)





NUM_FEATS = 24 # total number of CP child parameters

feats = np.zeros([c11.shape[0], c11.shape[1], NUM_FEATS])

s0 = c11 + c22
s1 = c11 - c22
s2 = 2 * c12_real
s3 = -2 * c12_imag

SV = np.zeros((s0.shape[0], s0.shape[1], 4))
SV[:,:,0] = s0
SV[:,:,1] = s1
SV[:,:,2] = s2
SV[:,:,3] = s3

# tiff.imwrite(p_directory + 'SV.tif', SV)

alpha_s = 0.5 * np.arctan2(np.sqrt(np.square(s1) + np.square(s2)) , s3)


nonzero_denom = np.where(s0+s3 != 0)
miu_c = np.zeros(c11.shape)
miu_c[nonzero_denom] = np.divide(s0[nonzero_denom]-s3[nonzero_denom] , s0[nonzero_denom]+s3[nonzero_denom])


nonzero_denom = np.where(s0 != 0)
u = np.zeros(c11.shape)
u[nonzero_denom] = np.divide(-s3[nonzero_denom] , s0[nonzero_denom])


nonzero_denom = np.where(s0 != 0)
rho = np.zeros(c11.shape)
rho[nonzero_denom] = 0.5 * np.sqrt(np.divide(np.square(s2[nonzero_denom])+np.square(s3[nonzero_denom]) , s0[nonzero_denom]))


delta = np.arctan2(s3 , s2)


nonzero_denom = np.where(s0 != 0)
m = np.zeros(c11.shape)
m[nonzero_denom] = np.divide(np.sqrt(np.square(s1[nonzero_denom])+np.square(s2[nonzero_denom])+np.square(s3[nonzero_denom])) , s0[nonzero_denom])


nonzero_denom = np.where(s0 != 0)
h_i = np.zeros(c11.shape)
h_i[nonzero_denom] = 2 * np.log10(math.pi * math.exp(1) * s0[nonzero_denom] / 2)


nonzero_denom = np.where((s0 != 0) & (np.square(s0) > np.square(s1)+np.square(s2)+np.square(s3)))
h_p = np.zeros(c11.shape)
h_p[nonzero_denom] = np.log10(np.divide(np.square(s0[nonzero_denom])-np.square(s1[nonzero_denom])-np.square(s2[nonzero_denom])-np.square(s3[nonzero_denom]) , np.square(s0[nonzero_denom])))


nonzero_denom = np.where(s0 != 0)
m_l = np.zeros(c11.shape)
m_l[nonzero_denom] = np.divide(np.sqrt(np.square(s1[nonzero_denom])+np.square(s2[nonzero_denom])) , s0[nonzero_denom])


nonzero_denom = np.where(s0+s1 != 0)
miu_l = np.zeros(c11.shape)
miu_l[nonzero_denom] = np.divide(s0[nonzero_denom]-s1[nonzero_denom] , s0[nonzero_denom]+s1[nonzero_denom])


psi = 0.5 * np.arctan2(s2 , s1)


nonzero_denom = np.where((s0 != 0) & (np.abs(s3) < np.abs(np.multiply(m, s0))))
r = np.zeros(c11.shape)
arc_sin_limit = np.where(np.abs(np.divide(s3[nonzero_denom] , np.multiply(m[nonzero_denom], s0[nonzero_denom]))) < 1)
r[nonzero_denom] = np.tan(0.5 * np.arcsin(np.divide(s3[nonzero_denom] , np.multiply(m[nonzero_denom], s0[nonzero_denom]))))



nonnegative = np.where(0.5 * (np.multiply(m , s0) + s3) >= 0)
mchi_b = np.zeros(c11.shape)
mchi_b[nonnegative] = np.sqrt(0.5 * (np.multiply(m[nonnegative] , s0[nonnegative]) + s3[nonnegative]))

nonnegative = np.where(0.5 * (np.multiply(m , s0) - s3) >= 0)
mchi_r = np.zeros(c11.shape)
mchi_r[nonnegative] = np.sqrt(0.5 * (np.multiply(m[nonnegative] , s0[nonnegative]) - s3[nonnegative]))

nonnegative = np.where(np.multiply(s0, 1-m) >= 0)
mchi_g = np.zeros(c11.shape)
mchi_g[nonnegative] = np.sqrt(np.multiply(s0[nonnegative], 1-m[nonnegative]))



nonnegative = np.where(np.multiply(0.5*np.multiply(s0 , m) , 1-np.sin(np.arctan2(s3,s2))) >= 0)
mdelta_r = np.zeros(c11.shape)
mdelta_r[nonnegative] = np.sqrt(np.multiply(0.5*np.multiply(s0[nonnegative] , m[nonnegative]) , 1-np.sin(np.arctan2(s3[nonnegative],s2[nonnegative]))))

nonnegative = np.where(np.multiply(s0, 1-m) >= 0)
mdelta_g = np.zeros(c11.shape)
mdelta_g[nonnegative] = np.sqrt(np.multiply(s0[nonnegative], 1-m[nonnegative]))

nonnegative = np.where(np.multiply(0.5*np.multiply(s0 , m) , 1+np.sin(np.arctan2(s3,s2))) >= 0)
mdelta_b = np.zeros(c11.shape)
mdelta_b[nonnegative] = np.sqrt(np.multiply(0.5*np.multiply(s0[nonnegative] , m[nonnegative]) , 1+np.sin(np.arctan2(s3[nonnegative],s2[nonnegative]))))

rh = c11
rv = c22



class features(Enum):
    ALPHA_S = 0
    MIU_C = 1
    U = 2
    RHO = 3
    DELTA = 4
    M = 5
    H_I = 6
    H_P = 7
    M_L = 8
    MIU_L = 9
    PSI = 10
    R = 11
    MCHI_B = 12
    MCHI_R = 13
    MCHI_G = 14
    MDELTA_R = 15
    MDELTA_G = 16
    MDELTA_B = 17
    RH = 18
    RV = 19
    S0 = 20
    S1 = 21
    S2 = 22
    S3 = 23


feat_directory = 'CP features from Python'
path = os.path.join(p_directory, feat_directory)
if(not os.path.isdir(path)):
   os.mkdir(path)

for f in features:
    if f.value == 0:
        feats[:, :, f.value] = alpha_s
        tiff.imsave(path + '/' + f.name + '.tif', alpha_s)
    elif f.value == 1:
        feats[:, :, f.value] = miu_c
        tiff.imsave(path + '/' + f.name + '.tif', miu_c)
    elif f.value == 2:
        feats[:, :, f.value] = u
        tiff.imsave(path + '/' + f.name + '.tif', u)
    elif f.value == 3:
        feats[:, :, f.value] = rho
        tiff.imsave(path + '/' + f.name + '.tif', rho)
    elif f.value == 4:
        feats[:, :, f.value] = delta
        tiff.imsave(path + '/' + f.name + '.tif', delta)
    elif f.value == 5:
        feats[:, :, f.value] = m
        tiff.imsave(path + '/' + f.name + '.tif', m)
    elif f.value == 6:
        feats[:, :, f.value] = h_i
        tiff.imsave(path + '/' + f.name + '.tif', h_i)
    elif f.value == 7:
        feats[:, :, f.value] = h_p
        tiff.imsave(path + '/' + f.name + '.tif', h_p)
    elif f.value == 8:
        feats[:, :, f.value] = m_l
        tiff.imsave(path + '/' + f.name + '.tif', m_l)
    elif f.value == 9:
        feats[:, :, f.value] = miu_l
        tiff.imsave(path + '/' + f.name + '.tif', miu_l)
    elif f.value == 10:
        feats[:, :, f.value] = psi
        tiff.imsave(path + '/' + f.name + '.tif', psi)
    elif f.value == 11:
        feats[:, :, f.value] = r
        tiff.imsave(path + '/' + f.name + '.tif', r)
    elif f.value == 12:
        feats[:, :, f.value] = mchi_b
        tiff.imsave(path + '/' + f.name + '.tif', mchi_b)
    elif f.value == 13:
        feats[:, :, f.value] = mchi_r
        tiff.imsave(path + '/' + f.name + '.tif', mchi_r)
    elif f.value == 14:
        feats[:, :, f.value] = mchi_g
        tiff.imsave(path + '/' + f.name + '.tif', mchi_g)
    elif f.value == 15:
        feats[:, :, f.value] = mdelta_r
        tiff.imsave(path + '/' + f.name + '.tif', mdelta_r)
    elif f.value == 16:
        feats[:, :, f.value] = mdelta_g
        tiff.imsave(path + '/' + f.name + '.tif', mdelta_g)
    elif f.value == 17:
        feats[:, :, f.value] = mdelta_b
        tiff.imsave(path + '/' + f.name + '.tif', mdelta_b)
    elif f.value == 18:
        feats[:, :, f.value] = rh
        tiff.imsave(path + '/' + f.name + '.tif', rh)
    elif f.value == 19:
        feats[:, :, f.value] = rv
        tiff.imsave(path + '/' + f.name + '.tif', rv)
    elif f.value == 20:
        feats[:, :, f.value] = s0
        tiff.imsave(path + '/' + f.name + '.tif', s0)
    elif f.value == 21:
        feats[:, :, f.value] = s1
        tiff.imsave(path + '/' + f.name + '.tif', s1)
    elif f.value == 22:
        feats[:, :, f.value] = s2
        tiff.imsave(path + '/' + f.name + '.tif', s2)
    elif f.value == 23:
        feats[:, :, f.value] = s3
        tiff.imsave(path + '/' + f.name + '.tif', s3)

    plt.imsave(path + '/' + f.name + '.png', feats[:, :, f.value], cmap='gray')

# feats = {'feats':feats}
# scipy.io.savemat(path + '/' + 'all_feats.mat', feats)


mchi = np.zeros((c11.shape[0], c11.shape[1], 3), dtype="uint8")
mchi[... ,0] = 255 * (mchi_r- np.min(mchi_r)) / (np.max(mchi_r) - np.min(mchi_r))
mchi[... ,1] = 255 * (mchi_g- np.min(mchi_g)) / (np.max(mchi_g) - np.min(mchi_g))
mchi[... ,2] = 255 * (mchi_b- np.min(mchi_b)) / (np.max(mchi_b) - np.min(mchi_b))
mchi_img = Image.fromarray(mchi)
mchi_img.save(path + '/' +'Mchi_RGB.png')

mdelta = np.zeros((c11.shape[0], c11.shape[1], 3), dtype="uint8")
mdelta[... ,0] = 255 * (mdelta_r- np.min(mdelta_r)) / (np.max(mdelta_r) - np.min(mdelta_r))
mdelta[... ,1] = 255 * (mdelta_g- np.min(mdelta_g)) / (np.max(mdelta_g) - np.min(mdelta_g))
mdelta[... ,2] = 255 * (mdelta_b- np.min(mdelta_b)) / (np.max(mdelta_b) - np.min(mdelta_b))
mdelta_img = Image.fromarray(mdelta)
mdelta_img.save(path + '/' +'Mdelta_RGB.png')

tiff.imwrite(path + '/feats.tif', feats, (feats.shape))
# to load you can just write:
#feats = tiff.imread(path + '/feats.tif')

