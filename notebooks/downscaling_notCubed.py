#Module contains Armaan's functions

import argparse
import pandas as pd
import numpy as np
import xarray as xr
import scipy
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import glob

## Importing Training Data


def import_HRDPS(year, variable):
    ##file paths
    data_name_hr = variable[0]
    files = glob.glob('/results/forcing/atmospheric/GEM2.5/gemlam/gemlam_y' + str(year) + 'm??d??.nc')
    files.sort()
    
    ## to accommodate the shift in time between the HRDPS data and the CanRCM data, we need to average across days.  So read all the data and then
    ## average to 3 hours.
    hrt = np.zeros( (24*len(files), 266, 256))
    hr = np.zeros( (8*len(files), 266, 256))
    
    for i in range(len(files)):
        dayX = xr.open_dataset(files[i])
        ##adding 1 day of 3-hour averages to new data array
        hrt[24*i:24*i + 24,:,:] = np.array( dayX[ data_name_hr ] )
        dayX.close()
    
    hr[:-1] = hrt[1:-2].reshape(8*len(files)-1, 3, 266, 256).mean(axis=1)  ## hrdrps is shifted one hour early, so first hour (0) is actually -1
                                                                   ## if I average, starting with hour 1 (which is really 0), I'll get an average
                                                                ## of 0-2 which is around 1
    hr[-1] = hrt[-2:].reshape(1, 2, 266, 256).mean(axis=1)  # one more at the end of the year based on only the last 2 values
    print ('hr shape', hr.shape)
    return hr

            
def import_CANRCM(year, variable):
    data_name_can = variable[1]
    p_can = '/data/sallen/results/CANRCM4/canrcm_' + data_name_can + '_' + str(year) + '.nc'
    ##CANRCM import
    d1 = xr.open_dataset(p_can)
    if year == 2007:
        can = d1[data_name_can][16:,140:165,60:85] ##the first two days are removed to be consistent with 2007 HRDPS
    elif year == 2008:
        can = np.concatenate((d1[data_name_can][:472,140:165,60:85], d1[data_name_can][464:472,140:165,60:85], d1[data_name_can][472:,140:165,60:85] ))
    else:
        can = d1[data_name_can][:,140:165,60:85]
    print ('can shape', can.shape)
    return can


def import_HRDPS_winds(year):
    
    ##file paths
    files = glob.glob('/results/forcing/atmospheric/GEM2.5/gemlam/gemlam_y' + str(year) + 'm??d??.nc')
    files.sort()
    ## calculating 3-hour averaged data
    hrt_u = np.zeros( (24*len(files), 266, 256)) 
    hrt_v = np.zeros( (24*len(files), 266, 256)) 
    hr_u = np.zeros( (8*len(files), 266, 256))
    hr_v = np.zeros( (8*len(files), 266, 256))

    for i in range(len(files)):
        dayX = xr.open_dataset(files[i])
        u = np.array( dayX['u_wind'] )
        v = np.array( dayX['v_wind'] )
        hrt_u[24*i:24*i + 24, : , : ] = u ##adding days to new data array
        hrt_v[24*i:24*i + 24, : , : ] = v
        dayX.close()
    
    ##averaging magnitudes/directions rather than averaging conponents
    ## do cubic average
    avg_spd = ((hrt_u[1:-2]**2 + hrt_v[1:-2]**2)**0.5).reshape(8*len(files)-1, 3, 266, 256).mean(axis=1)
    avg_th = np.arctan2(hrt_v[1:-2].reshape(8*len(files)-1, 3, 266, 256).mean(axis=1), hrt_u[1:-2].reshape(8*len(files)-1, 3, 266, 256).mean(axis=1))
    hr_u[:-1] = avg_spd*np.cos(avg_th)
    hr_v[:-1] = avg_spd*np.sin(avg_th) 
    # one more at the end based only the last two values
    avg_spd = ((hrt_u[-2:]**2 + hrt_v[-2:]**2)**0.5).reshape(1, 2, 266, 256).mean(axis=1)
    avg_th = np.arctan2(hrt_v[-2:].reshape(1, 2, 266, 256).mean(axis=1), hrt_u[-2:].reshape(1, 2, 266, 256).mean(axis=1))
    hr_u[-1] = avg_spd*np.cos(avg_th)
    hr_v[-1] = avg_spd*np.sin(avg_th) 
    return (hr_u, hr_v)


def import_CANRCM_winds(year):
    
    ##data paths
    p_can_u = '/data/sallen/results/CANRCM4/canrcm_uas_' + str(year) + '.nc'
    p_can_v = '/data/sallen/results/CANRCM4/canrcm_vas_' + str(year) + '.nc'
    ##importing data
    d1 = xr.open_dataset(p_can_u)
    d2 = xr.open_dataset(p_can_v)
    
    if year == 2007:
        can_u = d1['uas'][16:,140:165,60:85] ##the first two days are removed to be consistent with 2007 HRDPS
        can_v = d2['vas'][16:,140:165,60:85]
    elif year == 2008:
        can_u = np.concatenate((d1['uas'][:472,140:165,60:85], d1['uas'][464:472,140:165,60:85], d1['uas'][472:,140:165,60:85] ))
        can_v = np.concatenate((d2['vas'][:472,140:165,60:85], d2['vas'][464:472,140:165,60:85], d2['vas'][472:,140:165,60:85] ))
        ##repeating feb 28th twice for leap year
    else:
        can_u = d1['uas'][:,140:165,60:85]
        can_v = d2['vas'][:,140:165,60:85]
        
    avg_speed = (can_u**2 + can_v**2)**0.5
    avg_th = np.arctan2(can_v, can_u)
        
    return (avg_speed * np.cos(avg_th), avg_speed * np.sin(avg_th))


## CanRCM Weighting Function

def weight_CanRCM(can_solar, yr, start_day):
    yr_length = 365
    if yr == 2008:
        yr_length = 366
    size = yr_length * 8
    print (start_day, yr_length, size)
    newweighting = (300000/size-5000*np.cos(np.pi*(np.arange(size)+200)/size*2)/size*2*np.pi 
                    +2000*np.cos(np.pi*(np.arange(size)+200)/size*4)/size*4*np.pi)/100.
    # only use a fraction of the weighting
    fraction = 0.75
    newweighting = 1 + fraction * (newweighting - 1)
    lt, lx, ly = can_solar.shape[0], can_solar.shape[1], can_solar.shape[2]
    return can_solar * (np.broadcast_to(newweighting[(start_day-1)*8:], (ly, lx, lt)).transpose())


## PCA Functions

##data must be converted into a 2D matrix for pca analysis
##transform takes a 3D data array (time, a, b) -> (a*b, time)
##(the data grid is flattened a column using numpy.flatten)

def mytransform(xarr):
    arr = np.array(xarr) ##converting to numpy array
    arr = arr.reshape(arr.shape[0], arr.shape[1]*arr.shape[2]) ##reshaping from size (a, b, c) to (a, b*c)
    arr = arr.transpose()
    return arr


##transforms and concatenates two data sets
def transform2(data1, data2):
    A_mat = mytransform(data1)
    B_mat = mytransform(data2)
    return np.concatenate((A_mat, B_mat), axis=0) 


def reverse(mat, orig_shape):
    arr = np.copy(mat)
    arr = arr.transpose()
    arr = arr.reshape(-1, orig_shape[1], orig_shape[2]) ##reshaping back to original array shape
    return arr


##inverse function of transform2 - splits data matrix and returns two data sets
def reverse2(matrix, orig_shape):
    split4 = int( matrix.shape[0]/2 )
    u_data = reverse(matrix[:split4,:], orig_shape) ##reconstructing u_winds from n PCs
    v_data = reverse(matrix[split4:,:], orig_shape) ##reconstructing v_winds from n PCs
    return (u_data, v_data)


##performs PCA analysis using sklearn.pca

def doPCA(comp, matrix):
    pca = PCA(n_components = comp) ##adjust the number of principle conponents to be calculated
    PCs = pca.fit_transform(matrix)
    eigvecs = pca.components_
    mean = pca.mean_
    return (PCs, eigvecs, mean)


## Multiple Linear Regression

##functions that use multiple linear regression to fit eigenvectors
##takes CANRCM eigenvectors (x1, x2, x3, x4...) and HRDPS eigenvectors (y1, y2, y3...)
##For each y from 0:result_size, approximates yn = a0 + a1*x1 + a2*x2 + a3*x3 ... using num_vec x's
##getCoefs returns (coeficients, intercept)
##fit_modes returns each approximation and the R^2 value of each fit as (results, scores)

def getCoefs(vectors, num_vec, data, num_modes, type = 'LS'):  
    
    X = vectors[0:num_vec,:].T
    coefs = np.zeros((num_modes, X.shape[1]))
    intercept = np.zeros(num_modes)
    
    if type == 'LS':
        for i in range(num_modes):
            y = data[i,:]
            reg = LinearRegression().fit(X, y)
            coefs[i] = reg.coef_[0:num_vec]
            intercept[i] =  reg.intercept_
    elif type == 'MAE':
        for i in range(num_modes):
            y = data[i,:]
            reg = QuantileRegressor(quantile = 0.5, alpha = 0, solver = 'highs').fit(X, y)
            coefs[i] = reg.coef_[0:num_vec]
            intercept[i] =  reg.intercept_
    
    return (coefs, intercept)


def fit_modes(vectors, num_vec, data, result_size, type = 'LS'):  
    
    X = vectors[0:num_vec,:].T
    result = np.zeros((result_size, X.shape[0]))
    scores = np.zeros(result_size)
    
    if type == 'LS':
        for i in range(result_size):
            y = data[i,:]
            reg = LinearRegression().fit(X, y)
            result[i] = reg.predict(X)
            scores[i] = reg.score(X, y)
            
    elif type == 'MAE':
        for i in range(result_size):
            y = data[i,:]
            reg = QuantileRegressor(quantile = 0.5, alpha = 0, solver = 'highs').fit(X, y)
            result[i] = reg.predict(X)
            scores[i] = reg.score(X, y)
    
    return (result, scores)


## Projection Function

##scalar projection of u onto v - with extra 1/norm factor (for math reasons)
##projectData projects the data onto each principle conponent, at each time
##output is a set of eigenvectors

def project(u, v):  
    v_norm = np.sqrt(np.sum(v**2))    
    return np.dot(u, v)/v_norm**2


def projectData(data_mat, new_PCs, n):
    time = data_mat.shape[1]
    proj = np.empty((n, time))

    for j in range(n):
        for i in range(time):
            proj[j, i] = project(data_mat[:,i], new_PCs[:,j])
            
    return proj

## Overall Function

def reconstruct(downscale_mat, mean, can_PCs, can_me, hr_PCs, hr_me, n, r, method = 'LS', EB = 'False'):

    coefs = getCoefs(can_me, n + 1, hr_me, r + 1, type = method)
    proj = np.concatenate((mean.reshape(1, -1), projectData(downscale_mat - mean, can_PCs, n)), axis = 0)
    pred_eigs = np.matmul(coefs[0], proj) + coefs[1].reshape(-1, 1)  ##multiple linear regression output
    
    recon = np.matmul(hr_PCs[:,0:r], pred_eigs[1:r+1]) + pred_eigs[0]
    data_rec = reverse(recon, (-1, 266, 256))
    
    return data_rec

def reconstruct2(downscale_mat, mean, can_PCs, can_me, hr_PCs, hr_me, n, r, method = 'LS', EB = 'false'):

    coefs = getCoefs(can_me, n + 1, hr_me, r + 1, type = method)
    proj = np.concatenate((mean.reshape(1, -1), projectData(downscale_mat - mean, can_PCs, n)), axis = 0)
    pred_eigs = np.matmul(coefs[0], proj) + coefs[1].reshape(-1, 1)  ##multiple linear regression output
    
    recon = np.matmul(hr_PCs[:,0:r], pred_eigs[1:r+1]) + pred_eigs[0]
    u_data_rec, v_data_rec = reverse2(recon, (-1, 266, 256))
    
    return (u_data_rec, v_data_rec)


## Main Functions

##reconstructing other variables
def do_reconstruction(target_year, variables, data):
    for i in variables:

        data_name_hr = i[0]
        print (data_name_hr)

        can07 = import_CANRCM(2007, i)
        if data_name_hr == 'solar':
            can07 = weight_CanRCM(can07, 2007, 3)

        ##PCA on CANRCM 2007
        can07_mat = mytransform(can07)
        del can07
        can07_PCs, can07_eigs, can07_mean = doPCA(100, can07_mat)
        del can07_mat

        ##PCA on HRDPS 2007
        hr07 = import_HRDPS(2007, i)
        hr07_mat = mytransform(hr07)
        del hr07
        hr07_PCs, hr07_eigs, hr07_mean = doPCA(100, hr07_mat)
        del hr07_mat

        ## combining the eigenvectors and mean together in one array for analysis
        can07_me = np.concatenate((can07_mean.reshape(1, -1), can07_eigs))
        del can07_eigs
        hr07_me = np.concatenate((hr07_mean.reshape(1, -1), hr07_eigs))
        del hr07_eigs

        canTY = import_CANRCM(int(target_year), i)
        if target_year == '2007':
            start_day = 3
        else:
            start_day = 1
        print (f'Target Year = {target_year} and Start Day = {start_day}')
        
        if data_name_hr == 'solar':
            canTY = weight_CanRCM(canTY, int(target_year), start_day)
        canTY_mat = mytransform(canTY)
        del canTY
            
        mean_TY = canTY_mat.mean(axis = 0)
            
        data_rec = reconstruct(canTY_mat, mean_TY, can07_PCs, can07_me, hr07_PCs, hr07_me, 65, 65, method = 'LS')

        if data_name_hr == 'precip' or data_name_hr == 'qair' or data_name_hr == 'solar' or data_name_hr == 'therm_rad':
            avg = np.mean(data_rec, axis = 0)
            data_rec[data_rec < 0] = 0
            avg2 = np.mean(data_rec, axis = 0)
            data_rec *= avg/avg2

        data += ((data_name_hr, data_rec),)
        print(data_name_hr, "done")
    return data


def do_wind_reconstruction(target_year, data):
    ##reconstructing u and v winds
    
    can07_u, can07_v = import_CANRCM_winds(2007)
    print ('step 0')
    ##PCA on CANRCM 2007
    can07_mat = transform2(can07_u, can07_v)
    del can07_u
    del can07_v
    can07_PCs, can07_eigs, can07_mean = doPCA(100, can07_mat)
    del can07_mat
    print ('step 1')

    hr07_u, hr07_v = import_HRDPS_winds(2007)
    ##PCA on HRDPS 2007
    hr07_mat = transform2(hr07_u, hr07_v)
    del hr07_u
    del hr07_v
    hr07_PCs, hr07_eigs, hr07_mean = doPCA(100, hr07_mat)
    del hr07_mat
    print ('step 2')

    ## combining the eigenvectors and mean together in one array for analysis
    can07_me = np.concatenate((can07_mean.reshape(1, -1), can07_eigs))
    hr07_me = np.concatenate((hr07_mean.reshape(1, -1), hr07_eigs))
    del can07_eigs
    del hr07_eigs
    print ('step 3')

    ##calculating average of rows
    canTY_u, canTY_v = import_CANRCM_winds(int(target_year))
    print ('step 0')
    canTY_mat = transform2(canTY_u, canTY_v)
    del canTY_u
    del canTY_v
    mean_TY = canTY_mat.mean(axis = 0)

    u_data_rec, v_data_rec = reconstruct2(canTY_mat, mean_TY, can07_PCs, can07_me, hr07_PCs, hr07_me, 65, 65, method = 'LS')
    
    u_data_rec = 1.35 * u_data_rec
    v_data_rec = 1.35 * v_data_rec

    nspeed = (u_data_rec**2  + v_data_rec**2)**(0.5)
    nangle = np.arctan2(v_data_rec, u_data_rec)

    data += (('u_wind', nspeed * np.cos(nangle)),)
    data += (('v_wind', nspeed * np.sin(nangle)),)
    del nspeed
    del nangle
    print("u and v winds done")
             
    return data


# Write the Files
##creating netcdf files for each day and publishing them to the given path
##creates a netcdf file for each day

def write_the_files(data, stub, comments, start_date, end_date):
    data_var = {}
    dims = ('time_counter', 'y', 'x')
    times = np.arange(start_date, end_date, np.timedelta64(3, 'h'), dtype='datetime64[ns]')

    for i in range(int(times.shape[0]/8.)):
        for j in data:
            data_var[ j[0] ] = (dims, j[1][8*i:8*i + 8], {})
        coords = {'time_counter': times[8*i:8*i + 8], 'y': range(266), 'x': range(256)}
        ds = xr.Dataset(data_var, coords)
        ds.attrs['Comment'] = comments

        d = pd.to_datetime(times[8*i])
        filename = f'ncfiles/{stub}_y{d.year}m{d.month:02d}d{d.day:02d}.nc'

        encoding = {var: {'zlib': True} for var in ds.data_vars}
        ds.to_netcdf(filename, unlimited_dims=('time_counter'), encoding=encoding)
        print(f'# of files complete: {i+1}')
        
        
def get_avg_files(target_year):
    
    print (target_year)
    print ('Does cubed averaging still!!!')
    
    variables = [
            ['solar', 'rsds', 'Shortwave radiation'],
            ['tair', 'tas', 'Near-Surface Air Temperature'], 
            ['precip', 'pr', 'Precipitation'], 
            ['atmpres', 'psl', 'Sea Level Pressure'], 
            ['qair', 'huss', 'Near Surface Specific Humidity'], 
            ['therm_rad', 'rlds', 'Longwave radiation']]
    
    data = ()
    
    hr_u, hr_v = import_HRDPS_winds(int(target_year))
    nspeed = np.sqrt(hr_u**2  + hr_v**2)**(1/3)
    nangle = np.arctan2(hr_v, hr_u)

    data += (('u_wind', nspeed * np.cos(nangle)),)
    data += (('v_wind', nspeed * np.sin(nangle)),)
    del nspeed
    del nangle
    print("u and v winds done")
  
    
    for i in variables:

        data_name_hr = i[0]
        print (data_name_hr)
        hr07 = import_HRDPS(2007, i)
        data += ((data_name_hr, hr07),)
        print(data_name_hr, "done")
        
    start_day = '01'
    if target_year == '2007':
        start_day = '03'
    write_the_files(data, 'vectoravg', '3hr avg files, winds vector averaged', f'{target_year}-01-{start_day}T01:30', f'{int(target_year)+1}-01-01T01:30')
    
        
def main(target_year):
    
    print (target_year)
    
    variables = [
            ['solar', 'rsds', 'Shortwave radiation'],
            ['tair', 'tas', 'Near-Surface Air Temperature'], 
            ['precip', 'pr', 'Precipitation'], 
            ['atmpres', 'psl', 'Sea Level Pressure'], 
            ['qair', 'huss', 'Near Surface Specific Humidity'], 
            ['therm_rad', 'rlds', 'Longwave radiation']]
    
    data = ()
    data = do_wind_reconstruction(target_year, data)
    data = do_reconstruction(target_year, variables, data)
    start_day = '01'
    if target_year == '2007':
        start_day = '03'
    write_the_files(data, 'reconX', 'PCA vector winds', f'{target_year}-01-{start_day}T01:30', f'{int(target_year)+1}-01-01T01:30')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target_year', help='Year to create files for')
    parser.add_argument('get_raw_avg_files', help='Flag to get raw avg files, True or False')
    args = parser.parse_args()
    if args.get_raw_avg_files == 'True':
        get_avg_files(args.target_year)
    else:
        main(args.target_year)
