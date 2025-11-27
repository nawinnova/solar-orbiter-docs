"""
==================================
Finding and Plotting Metis Data
==================================

This example demonstrates how to search for, download, and plot 
Solar Orbiter Metis coronagraph observations using SunPy. This example is adapted from Metis tutorial notebooks avaliable at 
https://github.com/SolarOrbiterWorkshop/solo8_tutorials/tree/main/Metis_tutorial created by Aleksandr Burtovoi et al.

"""

import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.net import Fido, attrs as a
import sunpy_soar
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import skycoord_to_pixel



###############################################################################
# Registering the Solar Orbiter Archive (SOAR)
# --------------------------------------------
#
# By importing `sunpy_soar`, the Solar Orbiter Archive (SOAR) is **automatically**
# registered as a data provider in `Fido`. This allows us to directly query and 
# download Solar Orbiter data just like any other SunPy Fido search.
#
# Now, we will use `Fido` to search for available Metis Level 2 data. Metis L2 data are calibrated to be science-ready and are avaliable in two passbands: 
# visible light (VL) and ultraviolet Lyman-alpha (UV). The VL data are separated into two types: total brightness (tB) and polarized brightness (pB).
#
# More details about Metis data products can be found at http://metis.oato.inaf.it/data_products.html
#
# In this example, we will search for VL total brightness data. The search will return metadata about available files.

# valid search term for Metis products are: 'metis-vl-tb', 'metis-vl-pb', 'metis-uv-image'
search_results = Fido.search(a.Time("2023-04-09 07:30", "2023-04-09 08:30"),
                             a.soar.Product("metis-vl-tb"),
                             a.Level(2))

print(search_results)

###############################################################################
# Fetching the First Available File
# ---------------------------------
#
# The search results contain a list of available files matching our query.
# We can use `Fido.fetch` to download the first available file.

downloaded_files = Fido.fetch(search_results[0, 0])

###############################################################################
# Visualizing the Metis image with `sunpy.map`
# -------------------------------------------
#
# The downloaded file is in **FITS format**, which is commonly used for astronomical
# imaging data. We will use `sunpy.map.Map` to load and display it.
# However, in order to properly visualize the Metis data, we need to modify the metadata of FITS file, cut the inner and outer part of FOV, 
# and mask all the bad pixels.
#
# We will define two functions to do this: `cut_metis_fov` and `read_metis`

def cut_metis_fov(map, qflag, mask_value=np.nan):
    """
    Masks regions of the Metis instrument field of view (FOV) in the provided FITS HDU data array.
    This function sets pixels to NaN in the HDU data array based on their location relative to the instrument's
    inner and outer FOV boundaries, as well as a quality flag array. 
    Specifically:
    - Pixels within the inner occulter FOV (dist_iocen < fov1) are masked.
    - Pixels outside the outer FOV (dist_suncen > fov2) are masked.
    - Pixels where the quality flag is zero (qflag == 0) are masked.
    The FOV boundaries are calculated using header information from the HDU, including plate scale and FOV radii.
    Parameters
    ----------
    map : sunpy.map.Map
        SunPy map object containing the FITS HDU data and metadata.
    qflag : numpy.ndarray
        Array of quality flags with the same shape as the HDU data. Pixels with a value of 0 are considered invalid.
    mask_value : float, optional
        Value to use for masking invalid pixels. Default is NaN. 
        Use mask_value = 0 to mask with zeros, which works better for several image enhancement algorithms.
    Raises
    ------
    ValueError
        If the plate scale in the x and y directions (CDELT1 and CDELT2) are not equal.
    Returns
    -------
    map_new : sunpy.map.Map
        A new SunPy map object with the masked data.
    """

    # check if plate scale is the same in x and y direction
    if map.meta['CDELT1'] != map.meta['CDELT2']:
        raise ValueError("Error. CDELT1 != CDELT2 for {fname}".format(fname=map.meta['FILENAME']))
    # Get FOV in pixel
    fov1 = map.meta['INN_FOV']*3600/map.meta['CDELT1']  # pix
    fov2 = map.meta['OUT_FOV']*3600/map.meta['CDELT2']  # pix
    # Create meshgrid of pixel coordinates
    x = np.arange(0, map.meta['NAXIS1'], 1)
    y = np.arange(0, map.meta['NAXIS2'], 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # Calculate distance from Sun center and occulter center
    suncenter_pix = skycoord_to_pixel(SkyCoord(0*u.arcsec, 0*u.arcsec, frame=map.coordinate_frame), map.wcs)
    dist_suncen = np.sqrt((xx-suncenter_pix[0])**2 + (yy-suncenter_pix[1])**2)
    dist_iocen = np.sqrt((xx-map.meta['IO_XCEN'])**2 + (yy-map.meta['IO_YCEN'])**2)
    # Mask data outside FOV and bad pixels
    map.data[dist_iocen < fov1] = mask_value
    map.data[dist_suncen > fov2] = mask_value
    # Mask bad pixels based on quality flag
    map.data[qflag == 0] = mask_value

    map_new = sunpy.map.Map(map.data, map.meta)
    return map_new

def read_metis(filepath, rot=True):
    """
    Reads a Solar Orbiter Metis FITS file and returns a SunPy map object.
    This function opens the FITS file, processes the data to cut the field of view (FOV)
    and mark bad pixels, and then creates a SunPy map from the processed data.
    Parameters
    ----------
    filepath : str
        Path to the Metis FITS file.
    rot : bool, optional
        If True, the resulting SunPy map will be rotated to have solar north up. Default is True.
    Returns         
    -------
    map_metis: sunpy.map.Map
        A SunPy map object containing the processed Metis data.
    """
    hdu0 = fits.open(filepath)[0]
    qflag = fits.open(filepath)[1]
    # Update RSUN_OBS keyword to use RSUN_ARC value
    hdu0.header['RSUN_OBS'] = hdu0.header['RSUN_ARC']
    map_metis = sunpy.map.Map(hdu0.data, hdu0.header)

    map_metis = cut_metis_fov(map_metis, qflag.data)

    if rot == True:
        map_metis = map_metis.rotate()

    return map_metis

###############################################################################
# Now we can use the `read_metis` function to read the downloaded Metis FITS file
# and create a SunPy map. We can then plot the map using its built-in plot method.

metis_vl_tb_file = downloaded_files[0]
metis_vl_tb_map = read_metis(metis_vl_tb_file)

# Define a colormap that handles NaN values (bad pixels) - can be changed to any colormap
Metis_VL_CMAP = plt.get_cmap('afmhot').copy()
Metis_VL_CMAP.set_bad(color='tab:gray')  # np.nan values are in gray

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection=metis_vl_tb_map)
im = metis_vl_tb_map.plot(axes=ax, cmap=Metis_VL_CMAP)
metis_vl_tb_map.draw_limb(axes=ax, color='white', linewidth=1.0)
fig.colorbar(im, label='Mean Solar Brightness (MSB)')
ax.set_title('Metis VL tB '+ metis_vl_tb_map.date.strftime('%Y-%m-%d %H:%M:%S'))
plt.show()

###############################################################################
# Note that the time of observation is s/c time, which is different from Earth time.
# To get Earth time, access the 'date_ear' dict in the map metadata.



