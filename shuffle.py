#shuffle.py

#import necessary packages
from astropy.io import fits as f
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

def shuffle(cube_in,align,kind=None,interp=None):
     """
     Takes an input cube ('cube_in') and either a cube, velocity field, or value to
     rearrange the velocity axis of cube_in with respect to ('align'). the user must specify
     'kind' as either 'cube', 'vfield', or 'value'. if kind='cube', shuffle will generate a
     velocity field from the given cube. cube_in will then be shuffled onto a new velocity
     axis such that v=0 corresponds to the value of the genrated or given velocity field.
     if 'align' is a value, cube_in will be shifted by a constant value.
     interp specifies the type of interpolation to use when shifting spectra onto the new
     velocity axis. the options for the interpolation are 'linear', 'nearest', or 'cubic'
     (WARNING: cubic takes a while to execute)

     shuffle will return a file called cube_out.fits
     """
     
     if kind is None:
          kind='cube'

#read input cubes and headers
     hdus_in=f.open(cube_in)
     data_in=hdus_in[0].data
     header_in=hdus_in[0].header

     if (kind is 'cube') or (kind is 'vfield'):
          hdus_v=f.open(align)
          data_v=hdus_v[0].data
          header_v=hdus_v[0].header

#extract header info
     naxis_in=header_in['naxis']
     naxis1_in=header_in['naxis1']
     naxis2_in=header_in['naxis2']
     naxis3_in=header_in['naxis3']
     cdelt3_in=header_in['cdelt3']
     crpix3_in=header_in['crpix3']
     crval3_in=header_in['crval3']

     if (kind is 'cube') or (kind is 'vfield'):
          naxis_v=header_v['naxis']
          naxis1_v=header_v['naxis1']
          naxis2_v=header_v['naxis2']

#construct velocity axes and if necessary generate velocity field

     vaxis_in=((np.arange(naxis3_in)-(crpix3_in-1))*cdelt3_in+crval3_in)

     if kind is 'cube':
          naxis3_v=header_v['naxis3']
          cdelt3_v=header_v['cdelt3']
          crpix3_v=header_v['crpix3']
          crval3_v=header_v['crval3']

          vaxis_v=((np.arange(naxis3_v)-(crpix3_v-1))*cdelt3_v+crval3_v)

          mom1=vfield(data_v,vaxis_v)

     if kind is 'vfield':
          mom1=align

#define the new velocity axis as twice the old axis
     new_naxis=2*naxis3_in
     new_cdelt=cdelt3_in
     new_crpix=1
     new_crval=-(new_naxis/2)*new_cdelt

     new_vaxis=(np.arange(new_naxis)-(new_crpix-1))*new_cdelt+new_crval

#initialize the output cube
     grid=np.zeros(naxis1_in*naxis2_in*new_naxis).reshape(new_naxis,
                                                          naxis1_in,naxis2_in)
     hdu_out=f.PrimaryHDU(grid)
     hdus_out=f.HDUList([hdu_out])
     data_out=hdus_out[0].data
     header_out=hdus_out[0].header

#update the output header
     header_out=header_in
     header_out['naxis']=naxis_in
     header_out['naxis1']=naxis1_in
     header_out['naxis2']=naxis2_in
     header_out['naxis3']=new_naxis
     header_out['cdelt3']=new_cdelt
     header_out['crpix3']=new_crpix
     header_out['crval3']=new_crval

#begin a loop over the cube to shift the spectrum at each point onto new_vaxis

     for i in np.arange(naxis1_in):
          for j in np.arange(naxis2_in):
               if np.isfinite(mom1[i,j])==False:
                    continue
              
               if kind is 'value':
                    vc=align

               if (kind is 'cube') or (kind is 'vfield'):
                    vc=mom1[i,j]

               this_spec=data_in[:,i,j]
               
               data_out[:,i,j]=shift(this_spec,vaxis_in,new_vaxis,vc,
                                     interp)

#save and write output fits file
     hdus_out[0].data=data_out
     hdus_out[0].header=header_out
     hdus_out.writeto('cube_out.fits')

def shift(this_spec,vaxis_in,new_vaxis,vc,interp=None):
     """
     Function to shift a given spectrum to a new velocity axis.
     the interpolation can be linear, nearest, or cubic
     """

     this_vaxis=vaxis_in-vc

     orig_chan=np.arange(this_vaxis.size)
     n_chan=orig_chan.size

     if (interp is None) or (interp is 'linear'):
          linear_interp=interpolate.interp1d(this_vaxis,this_spec,
                                             bounds_error=False,
                                             fill_value=np.nan)

     if interp is 'nearest':
          linear_interp=interpolate.interp1d(this_vaxis,this_spec,
                                             bounds_error=False,
                                             fill_value=np.nan,
                                             kind='nearest')

     if interp is 'cubic':
          linear_interp=interpolate.interp1d(this_vaxis,this_spec,
                                             bounds_error=False,
                                             fill_value=np.nan,
                                             kind='cubic')

     new_spec=linear_interp(new_vaxis)

     return new_spec

def vfield(data_v,vaxis):
     """
     Generates a velocity field from the input data. invalid data points
     are filled with nans. A median filter is applied (this should eventually
     be a user defined option).
     """

     data_no_nans=data_v*1.0
     data_no_nans[np.isfinite(data_no_nans)==False]=0
     mom=np.sum(data_no_nans,axis=0)
     mom1=data_no_nans[0,:,:]*0.0
     for i in np.arange(mom[0,:].size):
          for j in np.arange(mom[:,0].size):
               if mom[i,j]==0:
                    mom1[i,j]=np.nan
                    continue
               mom1[i,j]=np.dot(vaxis,data_v[:,i,j])/mom[i,j]

     mom1=sig.medfilt(mom1,7)

     return mom1
