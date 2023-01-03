from datetime import datetime
import cftime
import cf_units
import numpy as np
import xarray as xr
from xhistogram.xarray import histogram
import pop_tools

def _get_tb_name_and_tb_dim(ds):
    """return the name of the time 'bounds' variable and its second dimension"""
    assert "bounds" in ds.time.attrs, 'missing "bounds" attr on time'
    tb_name = ds.time.attrs["bounds"]
    assert tb_name in ds, f'missing "{tb_name}"'
    tb_dim = ds[tb_name].dims[-1]
    return tb_name, tb_dim


def center_time(ds):
    """make time the center of the time bounds"""
    ds = ds.copy()
    attrs = ds.time.attrs
    encoding = ds.time.encoding
    tb_name, tb_dim = _get_tb_name_and_tb_dim(ds)
    new_time = ds[tb_name].compute().mean(tb_dim).squeeze()
    # Make sure it is a list if length 1
    if len(ds[tb_name]) == 1:
        new_time = [new_time.values]
    ds['time'] = new_time
    attrs["note"] = f"time recomputed as {tb_name}.mean({tb_dim})"
    ds.time.attrs = attrs
    ds.time.encoding = encoding
    return ds

def global_mean(ds, horizontal_dims, area_field, land_sea_mask, normalize=True, include_ms=False, region_mask=None, time_dim="year"):
    """
    Compute the global mean on some dataset
    Return computed quantity in conventional units.
    """

    compute_vars = [
        v for v in ds if time_dim in ds[v].dims and horizontal_dims == ds[v].dims[-2:]
    ]

    other_vars = list(set(ds.variables) - set(compute_vars))

    if include_ms:
        surface_mask = ds[area_field].where(ds[land_sea_mask] > 0).fillna(0.0)
    else:
        surface_mask = ds[area_field].where(ds.REGION_MASK > 0).fillna(0.0)

    if region_mask is not None:
        surface_mask = surface_mask * region_mask

    masked_area = {
        v: surface_mask.where(ds[v].notnull()).fillna(0.0) for v in compute_vars
    }

    with xr.set_options(keep_attrs=True):

        dso = xr.Dataset(
            {v: (ds[v] * masked_area[v]).sum(horizontal_dims) for v in compute_vars}
        )
        if normalize:
            dso = xr.Dataset(
                {v: dso[v] / masked_area[v].sum(horizontal_dims) for v in compute_vars}
            )

        return xr.merge([dso, ds[other_vars]]).drop(
            [c for c in ds.coords if ds[c].dims == horizontal_dims]
        )

def zonal_mean(da_in, grid, lat_axis=None, lat_field='geolat', ydim='yh', xdim='xh', area_field='area_t', region_mask=None):
    """Calculate a zonal average from some model on xarray.DataArray
    
    Parameters
    ----------
    
    da_in : xarray.DataArray
       DataArray to calculate a zonal average from. This should be your data variable
       
    grid : xarray.Dataset
       Grid with the latitude, area field, and latitude axis (if needed), matching dims of da_in
       
    lat_axis : xarray.DataArray
       Latitude axis to use for latitude bins
    
    lat_field : string
       Name of the latitude field to use
    
    ydim : string
       Name of y-dimension
    
    xdim : string
       Name of x-dimension
       
    area_field : string
       Field to use for the area values, used for weighting
       
    Returns
    -------
    da_out : xarray.DataArray
       Resultant zonally averaged field, with the same input name and a new latitude bin axis
    """

    # If not provided a latitude axis, use the y-axis
    if lat_axis is None:
        lat_axis = grid[ydim]
    
    area = grid[area_field].broadcast_like(da_in).where(da_in > -9999)
    lat_2d = grid[lat_field]
    
    if region_mask is not None:
        da_in = da_in.where(region_mask>0)
        area = area * region_mask.where(region_mask>0)
        lat_2d = lat_2d.where(region_mask>0)
    
    # Create the latitude bins using the lat_axis data array
    bins =  lat_axis.values
    
    # Calculate the numerator
    histVolCoordDepth = histogram(lat_2d.broadcast_like(area).where(~np.isnan(area)), bins=[bins], weights=area, dim=[ydim, xdim])
    
    # Calculate the denominator
    histTVolCoordDepth = histogram(lat_2d.broadcast_like(area).where(~np.isnan(area)), bins=[bins], weights=(area*da_in).fillna(0), dim=[ydim, xdim])
    
    if region_mask is not None:
        histRegionVolCoordDepth = histogram(lat_2d.broadcast_like(area).where(~np.isnan(area)), bins=[bins], weights=(area*region_mask).fillna(0), dim=[ydim, xdim])
    
    da_out = (histTVolCoordDepth/histVolCoordDepth).rename(da_in.name)
    
    # Return the zonal average, renaming the variable to the variable in
    return da_out


def regional_zonal_mean(da_in, grid, rmask, rmaskdict, lat_axis, lat_field='TLAT', ydim='nlat', xdim='nlon', area_field='TAREA'):
    """Calculate a zonal average from some model on xarray.DataArray. Must provide a region_mask which will define
    leading dimension of output.
    
    Input
    ----------
    da_in : xarray.DataArray
       DataArray to calculate a zonal average from. This should be your data variable
    grid : xarray.Dataset
       Grid with the latitude, area field, and latitude axis (if needed), matching dims of da_in
    rmask: xarray.DataArray
       DataArray containing region mask information (integers>0)
    rmaskdict: dictionary
       Dictionary that relates region mask values to region descriptions
    lat_axis : xarray.DataArray
       Latitude axis to use for latitude bins
    lat_field : string
       Name of the latitude field to use
    ydim : string
       Name of y-dimension
    xdim : string
       Name of x-dimension
    area_field : string
       Field to use for the area values, used for weighting
       
    Returns
    -------
    da_out : xarray.DataArray
       Resultant zonally averaged field, with the same input name and a new latitude bin axis
    """
    
    #area = grid[area_field].broadcast_like(da_in).where(da_in > -9999)
    area = xr.ones_like(da_in)*grid[area_field].where(~np.isnan(da_in))
    lat_2d = grid[lat_field]
    bins =  lat_axis.values
    zmlist = []
    # Iterate over region mask regions:
    for i in rmaskdict:
        if i==0:
            da_masked = da_in.where(rmask>0)
            area_masked = area.where(rmask>0)
            lat_2d_masked = lat_2d.where(rmask>0)
        else:
            da_masked = da_in.where(rmask==i)
            area_masked = area.where(rmask==i)
            lat_2d_masked = lat_2d.where(rmask==i)
        
        # Calculate the numerator
        lat_2d_masked = (xr.ones_like(area_masked)*lat_2d_masked).where(~np.isnan(area_masked))
        lat_2d_masked.name = lat_2d.name
        histVolCoordDepth = histogram(lat_2d_masked, bins=[bins], weights=area_masked, dim=[ydim, xdim])
        # Calculate the denominator
        histTVolCoordDepth = histogram(lat_2d_masked, bins=[bins], weights=(area_masked*da_masked).fillna(0), dim=[ydim, xdim])
        zm = (histTVolCoordDepth/histVolCoordDepth).rename(da_in.name)
        zm = zm.assign_coords({'region':rmaskdict[i]})
        zmlist.append(zm)
    da_out = xr.concat(zmlist,dim='region')
    
    return da_out

def time_set_mid(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with midpoints of
    ds[time_name].attrs['bounds'], if bounds attribute exists.
    Except for time_name, the returned Dataset is a copy of ds2.
    The copy is deep or not depending on the argument deep.
    """

    ds_out = ds.copy(deep)

    if "bounds" not in ds[time_name].attrs:
        return ds_out

    tb_name = ds[time_name].attrs["bounds"]
    tb = ds[tb_name]
    bounds_dim = next(dim for dim in tb.dims if dim != time_name)

    # Use da = da.copy(data=...), in order to preserve attributes and encoding.

    # If tb is an array of datetime objects then encode time before averaging.
    # Do this because computing the mean on datetime objects with xarray fails
    # if the time span is 293 or more years.
    #     https://github.com/klindsay28/CESM2_coup_carb_cycle_JAMES/issues/7
    if tb.dtype == np.dtype("O"):
        units = "days since 0001-01-01"
        calendar = "noleap"
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
        tb_mid_decode = cftime.num2date(
            tb_vals.mean(axis=1), units=units, calendar=calendar
        )
        ds_out[time_name] = ds[time_name].copy(data=tb_mid_decode)
    else:
        ds_out[time_name] = ds[time_name].copy(data=tb.mean(bounds_dim))

    return ds_out

def mon_to_seas(ds):
    """ Converts a DataSet containing monthly data to one containing 
    seasonal-average data. 
    """
    ds_seas = ds.rolling(time=3,min_periods=3, center=True).mean()
    time = ds_seas.time
    mon = time.dt.month.values
    keep = time.where((mon == 1) | (mon == 4) | (mon == 7) | (mon == 10)).dropna('time')
    ds_seas = ds_seas.sel(time=keep)
    return ds_seas

def mon_to_seas2(da):
    """ Converts a DataArray containing monthly data to one containing 
    seasonal-average data. 
    """
    seas_coord = xr.DataArray(['DJF','MAM','JJA','SON'],dims='season',name='season')
    da_roll = da.rolling(time=3,min_periods=3, center=True).mean()
    time = da_roll.time
    mon = time.dt.month.values
    dalist = []
    for month in [1,4,7,10]:
        keeptime = time.where((mon == month)).dropna('time')
        da_tmp = da_roll.sel(time=keeptime)
        da_tmp['time'] = da_tmp.time.dt.year
        dalist.append(da_tmp)
    ds_seas = xr.concat(dalist,dim=seas_coord)
    return ds_seas
