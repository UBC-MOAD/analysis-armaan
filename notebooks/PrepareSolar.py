# Postprocess Solar

import argparse
import arrow
import numpy as np
import pandas as pd
import xarray as xr

vsmall = 1e-4

def gen_solar(hour, day, latitude, longitude, offset):
    
    hour = hour + offset
    if hour > 24:
        day = day + 1
        hour = hour - 24
    elif hour < 0:
        day = day - 1
        hour = hour + 24
    
    ST = 24/360.*(longitude - (-120))  ## difference between actual longitude and central longitude
    day_time = ((hour - 8 - ST)%24 )*3600 ##converting 3 hourly UTC to seconds from midnight PST
       
    hour = (day_time / 3600.0 - 12.0) * 15.0  ##degrees
    declination = 23.45*np.pi/180.0*np.sin((284.0 + day)/365.25*2.0*np.pi)  ##radians
    
    ## Convert latitude from degrees to radians
    lat = np.pi * latitude / 180.0
    
    a = np.sin(declination) * np.sin(lat) 
    b = np.cos(declination) * np.cos(lat)
    cos_Z = a + b * np.cos(np.pi / 180.0 * hour)      ##solar elevation
    hour_angle = np.tan(lat)*np.tan(declination)  ## cos of -hour_angle in radians
    day_length =  np.arccos(-hour_angle) / 15.0 * 2.0 * 180.0 / np.pi ## hours: 15 = 360/24
    day_length[hour_angle > 1] = 24 + 0.0000001  ## so far North in summer that there is no night
    day_length[hour_angle < -1] = -0.0000001 ## so far North in winter that there is no day
        
    sunrise = 12.0 - 0.5 * day_length  ##hours
    sunset = 12.0 + 0.5 * day_length   ##hours
    
    Q_o = 1368.0     ## W/m^2 Solar constant
    Qso = Q_o*(1.0+0.033*np.cos(day/365.25*2.0*np.pi))
    
    I_incident = Qso * cos_Z
    I_incident[day_time/3600. < sunrise] = 0.
    I_incident[day_time/3600. > sunset] = 0.
    
    
    
    return I_incident

# need to average solar over same interval as you use it.  Make it an xarray database at say 10 minutes to make that eacy


def calculate_max_timeseries(year, month, day, lats, lons):
    ny, nx = lats.shape
    starttime = arrow.get(year, month, day, 0, 0, 0) 
    endtime = starttime.shift(days=+1)
    deltat = arrow.get(year, month, day, 0, 10, 0) - arrow.get(year, month, day, 0, 0, 0)
    solar = np.zeros((int(86400/10/60), ny, nx))
    times = np.zeros(solar.shape[0]).astype('datetime64[s]')
    time = starttime
    ii = 0
    while time < endtime:
        hour = time.hour + time.minute/60.
        yearday = int(time.format('DDDD'))
        solar[ii] = gen_solar(hour, yearday, lats, lons, offset=0)
        times[ii] = time.naive
        ii = ii + 1
        time = time + deltat
        
    da_solar = xr.DataArray(
        data=solar,
        dims=['time_counter', 'y', 'x'],
        coords=dict(
            time_counter=times,),

        attrs=dict(
            description="Max Solar",
            units="W/m2",),)
    
    maxsolar_1h = da_solar.resample(time_counter='1h', loffset=('30min')).mean() 
    maxsolar_3h = da_solar.resample(time_counter='3h', loffset=('90min')).mean()

    return maxsolar_1h, maxsolar_3h


def make_one_hour_series(threehour, maxsolar_3h, maxsolar_1h):
    
    ratio = np.array(threehour.solar[:, :, :]) / (np.array(maxsolar_3h[0:8, :, :]) + vsmall)
    ratio[ratio > 1] = 1
    
    myvalues = np.empty_like(maxsolar_1h)
    myvalues[0::3] = ratio * maxsolar_1h[0::3]
    myvalues[1::3] = ratio * maxsolar_1h[1::3]
    myvalues[2::3] = ratio * maxsolar_1h[2::3]
    
    return myvalues


def write_the_file(myvalues, times, stub, comments, year, month, day):
    
    ny, nx = myvalues[0].shape
    
    coords = {'time_counter': times, 'y': range(ny), 'x': range(nx)}
    
    df_solar = xr.Dataset(
        data_vars=dict(solar=(['time_counter', 'y', 'x'], myvalues)),
        coords=coords,
        attrs=dict(
            description="Solar",
            units="W/m2",
           Comment= comments ),)

    filename = f'ncfiles/{stub}_y{year}m{month:02d}d{day:02d}.nc'

    encoding = {var: {'zlib': True} for var in df_solar.data_vars}
    df_solar.to_netcdf(filename, unlimited_dims=('time_counter'), encoding=encoding)
        
    
def main_driver(year, stubin, stubout):
    # get lats and lons
    basefile = xr.open_dataset(f'/results/forcing/atmospheric/GEM2.5/gemlam/gemlam_y2007m03d01.nc')
    lons, lats = basefile.nav_lon.values, basefile.nav_lat.values
    basefile.close()
    
    startdate = arrow.get(year, 1, 1, 0, 0, 0) # to match hrdps, see below
    enddate = startdate.shift(years=1)
    enddate = enddate.shift(days=-1)
    deltaday = arrow.get(year, 1, 2, 0, 0, 0) - arrow.get(year, 1, 1, 0, 0, 0)
    time = startdate
    
    while time < enddate:
        
        maxsolar_1h, maxsolar_3h = calculate_max_timeseries(time.year, time.month, time.day, 
                                                                   lats, lons)
    
        threehour = xr.open_dataset(f'ncfiles/{stubin}_y{time.year}m{time.month:02d}d{time.day:02d}.nc')
        
        myvalues = make_one_hour_series(threehour, maxsolar_3h, maxsolar_1h)
        
        write_the_file(myvalues, maxsolar_1h.time_counter, stubout, 
                       'Hourly based on max and 3 hour time series', 
                       time.year, time.month, time.day)
        
        if time.day == 15:
            print (time)
        
        time = time + deltaday
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target_year', help='Year to create files for')
    parser.add_argument('stubin', help='Stub of name of 3 hour files')
    parser.add_argument('stubout', help='Stub of name of 1 hour files')
    args = parser.parse_args()
    main_driver(int(args.target_year), args.stubin, args.stubout)
