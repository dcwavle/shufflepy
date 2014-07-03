#shuffle.py
#takes two input data cubes, cube_in and cube_v. it generates
#a velocity field from cube_v and rearanges the velocity axis of cube_in
#so that v=0 corresponds to the local velocity of cube_v.

def shuffle(cube_in,cube_v):
     
#import necessary packages
     from astropy.io import fits as f
     import numpy as np
     import matplotlib.pyplot as plt
     import scipy.signal as sig
     import scipy.interpolate as interpolate
     import matplotlib.pyplot as plt

#read input cubes and headers
     hdus_in=f.open(cube_in)
     data_in=hdus_in[0].data
     header_in=hdus_in[0].header

     hdus_v=f.open(cube_v)
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

     naxis_v=header_v['naxis']
     naxis1_v=header_v['naxis1']
     naxis2_v=header_v['naxis2']
     naxis3_v=header_v['naxis3']
     cdelt3_v=header_v['cdelt3']
     crpix3_v=header_v['crpix3']
     crval3_v=header_v['crval3']

#slice out stokes axes
     if naxis_in>3:
          data_in=data_in[0,:,:,:]
          naxis_in=3
          del header_in['naxis4']

     if naxis_v>3:
          data_v=data_v[0,:,:,:]
          naxis_v=3

#slice an input cube if it is larger than the other (assuming they are
# on the same scale and the smaller cube has been sliced symmetrically)
     if naxis1_in>naxis1_v:
          s1=(naxis1_in/2-naxis1_v/2)
          s2=(naxis1_in/2+naxis1_v/2)
          s3=(naxis2_in/2-naxis2_v/2)
          s4=(naxis2_in/2+naxis2_v/2)
          data_in=data_in[:,s1:s2,s3:s4]
          naxis1_in=data_in[0,0,:].size
          naxis2_in=data_in[0,:,0].size

     if naxis1_in<naxis1_v:
          s1=(naxis1_v/2-naxis1_in/2)
          s2=(naxis1_v/2+naxis1_in/2)
          s3=(naxis2_v/2-naxis2_in/2)
          s4=(naxis2_v/2+naxis2_in/2)
          data_v=data_v[:,s1:s2,s3:s4]
          naxis1_v=data_v[0,0,:].size
          naxis2_v=data_v[0,:,0].size

#construct velocity axes
     vaxis_in=((np.arange(naxis3_in)-(crpix3_in-1))*cdelt3_in+crval3_in)
     vaxis_v=((np.arange(naxis3_v)-(crpix3_v-1))*cdelt3_v+crval3_v)

#generate velocity field
     data_no_nans=data_v*1.0
     data_no_nans[np.isfinite(data_no_nans)==False]=0
     mom=np.sum(data_no_nans,axis=0)
     mom1=np.zeros(mom.size).reshape(naxis1_v,naxis2_v)
     for i in np.arange(naxis1_v):
          for j in np.arange(naxis2_v):
               if mom[i,j]==0:
                    mom1[i,j]=np.nan
                    continue
               mom1[i,j]=np.dot(vaxis_v,data_v[:,i,j])/mom[i,j]

     mom1=sig.medfilt(mom1,7)

#get the original channel axis and number of original channels
     orig_chan=np.arange(vaxis_in.size)
     n_chan=orig_chan.size

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

#Define a function to shift a given spectrum to new_vaxis using a linear
#interpolation
     def shift(x,y):
          this_vaxis=vaxis_in-mom1[x,y]
          this_spec=data_in[:,x,y]

          linear_interp=interpolate.interp1d(this_vaxis,orig_chan,
                                                  bounds_error=False,
                                                  fill_value=0)
          sample_chan=linear_interp(new_vaxis)
          interp_ind=np.where((sample_chan>0)&(sample_chan<(n_chan-1)))
          interp_ct=np.where((sample_chan<=0)&(sample_chan>=(n_chan-1)))

          chan_hi=np.ceil(sample_chan[interp_ind]).astype(int)
          chan_lo=np.floor(sample_chan[interp_ind]).astype(int)

          slope=this_spec[chan_hi]-this_spec[chan_lo]
          offset=sample_chan[interp_ind]-chan_lo
          new_spec=np.zeros(new_naxis)
          new_spec[interp_ind]=this_spec[chan_lo]+slope*offset

          eq_ind=np.where(np.equal(chan_hi,chan_lo))
          if np.any(np.isfinite(eq_ind))==True:
               new_spec[interp_ind[eq_ind]]=this_spec[chan_lo[eq_ind]]

          return new_spec


     for i in np.arange(naxis1_in):
          for j in np.arange(naxis2_in):
               if np.isfinite(mom1[i,j])==False:
                    continue
               if mom1[i,j]==0:
                    continue
               data_out[:,i,j]=shift(i,j)

#save and write output fits file
     hdus_out[0].data=data_out
     hdus_out[0].header=header_out
     hdus_out.writeto('cube_out.fits')
