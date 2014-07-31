#shuffle.py

#import necessary packages
from astropy.io import fits as f
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import sys


def function(cube_in,align,kind=None,interp=None,define_axis=False):
     """
     Takes an input cube ('cube_in') and either a cube, velocity field, or value to
     rearrange the velocity axis of cube_in with respect to ('align'). the user can specify
     'kind' as either 'cube', 'vfield', or 'value'. if kind='cube'(default), shuffle will generate a
     velocity field from the given cube. cube_in will then be shuffled onto a new velocity
     axis such that v=0 corresponds to the value of the genrated or given velocity field or value.
     interp specifies the type of interpolation to use when shifting spectra onto the new
     velocity axis. the options for the interpolation are 'linear' (default), 'nearest', or 'cubic'
     (WARNING: cubic takes a while to execute)

     shuffle will return a file called cube_out.fits
     """
     
     if kind is None:
          kind='cube'

#read input cubes and headers and grab header info:
     (data_in,hdr_in,naxis_in,naxis1_in,naxis2_in,naxis3_in,cdelt3_in,
      crpix3_in,crval3_in)=read(cube_in,kind='cube')

     if kind is 'cube':
          (data_v,hdr_v,naxis_v,naxis1_v,naxis2_v,naxis3_v,cdelt3_v,
           crpix3_v,crval3_v)=read(align,kind=kind)

     if kind is 'vfield':
          (data_v,hdr_v,naxis_v,naxis1_v,naxis2_v)=read(align,kind=kind) 

#do some error checking

     if naxis_in>3:
          sys.exit("input cube must have 3 dimensions")

     if kind is 'cube' or kind is 'vfield':
          if naxis1_in != naxis1_v or naxis2_in != naxis2_v:
               sys.exit("inputs must have the same spatial dimensions")

     if kind is 'vfield':
          if naxis_v>2:
               sys.exit("vfield must have 2 dimensions")       

#construct velocity axes and if necessary generate velocity field

     vaxis_in=make_axis(hdr_in)

     if kind is 'cube':
          vaxis_v=((np.arange(naxis3_v)-(crpix3_v-1))*cdelt3_v+crval3_v)
          align=vfield(data_v,vaxis_v)

#define the new velocity axis. default new_axis is  twice the old axis with half the 
#velocity step size

     if define_axis==False:
          new_naxis=4*naxis3_in
          new_cdelt=cdelt3_in/2
          new_crpix=1
          new_crval=-(new_naxis/2)*new_cdelt

     if define_axis==True:
          new_naxis=raw_input("enter value for new_naxis: ")
          new_cdelt=raw_input("enter value for new_cdelt: ")
          new_crpix=raw_input("enter value for new_crpix: ")
          new_crval=raw_input("enter value for new_crval: ")

     new_vaxis=make_axis(naxis=new_naxis,cdelt=new_cdelt,crpix=new_crpix,
                crval=new_crval)

#shuffle the cube

     data_out=shuffle(data_in,align,vaxis_in,new_vaxis,kind=kind,interp=interp)

#write to an output fits file 

     hdr_out=hdr_in
     hdr_out['naxis3']=new_naxis
     hdr_out['cdelt3']=new_cdelt
     hdr_out['crpix3']=new_crpix
     hdr_out['crval3']=new_crval

     hdu_out=f.PrimaryHDU(data_out)
     hdus_out=f.HDUList([hdu_out])
     hdus_out[0].header=hdr_out

     hdus_out.writeto('cube_out.fits')

def make_axis(hdr=None,naxis=None,crpix=None,cdelt=None,crval=None):

     """
     Constructs velocity axis. if hdr=true (default) grabs axis parameters from FITS header. if
     hdr=False the user is promted for axis parameters.
     """

     if hdr and (naxis or crpix or cdelt or crval):
          sys.exit("please input header or specify header parameters, not both")

     if hdr:
          naxis=hdr['naxis3']
          crpix=hdr['crpix3']
          cdelt=hdr['cdelt3']
          crval=hdr['crval3']

     vaxis=((np.arange(naxis)-(crpix-1))*cdelt+crval)

     return vaxis

def read(fits_file,kind=None):
     """
     reads an input fits file. only returns values that 
     are relevant to the shuffle program
     """

     hdus=f.open(fits_file)
     data=hdus[0].data
     hdr=hdus[0].header

     naxis=hdr['naxis']
     naxis1=hdr['naxis1']
     naxis2=hdr['naxis2']

     if kind is 'cube':
          naxis3=hdr['naxis3']
          cdelt3=hdr['cdelt3']
          crpix3=hdr['crpix3']
          crval3=hdr['crval3']

     if kind is 'cube':
          return data,hdr,naxis,naxis1,naxis2,naxis3,cdelt3,crpix3,crval3

     if kind is 'vfield':
          return data,hdr,naxis,naxis1,naxis2

def shuffle(data_in,align,vaxis_in,new_vaxis,kind=None,interp=None):
     """
     takes and input array ("data_in") and a corresponding array, map, or value to
     rearrange the velocity axis of "data_in" with respect to. Specify the type of 
     "align" with kind='cube', 'vfield', or 'value'. if "align" is an array, shuffle 
     will generate a moment map to align "data_in" to. "data_in" must be a 3d array.
     """

     if interp is None:
          interp='linear'

     sz1=data_in[0,0,:].size
     sz2=data_in[0,:,0].size
     sz3=new_vaxis.size

     data_out=np.zeros(sz1*sz2*sz3).reshape(sz3,sz2,sz1)

     for i in np.arange(data_in[0,0,:].size):
          for j in np.arange(data_in[0,:,0].size):
               if np.isfinite(align[i,j])==False:
                    for k in np.arange(new_vaxis.size):
                         data_out[k,i,j]=np.nan
                    continue

               if kind is 'value':
                    vc=align

               if (kind is 'cube') or (kind is 'vfield'):
                    vc=align[i,j]

               this_spec=data_in[:,i,j]
               
               data_out[:,i,j]=shift(this_spec,vaxis_in,new_vaxis,vc,
                                     interp)

     return data_out

def shift(this_spec,vaxis_in,new_vaxis,vc,interp=None):
     """
     Function to shift a given spectrum to a new velocity axis.
     the interpolation can be linear, nearest, or cubic
     """

     this_vaxis=vaxis_in-vc

     orig_chan=np.arange(this_vaxis.size)
     n_chan=orig_chan.size

     if (interp is None):
          interp='linear'

     interpolation=interpolate.interp1d(this_vaxis,this_spec,
                                             bounds_error=False,
                                             fill_value=np.nan,
                                             kind=interp)

     new_spec=interpolation(new_vaxis)

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
