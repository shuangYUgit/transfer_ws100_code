import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import netCDF4 as nc
import os
from global_land_mask import globe
from sklearn.linear_model import Lasso
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
plt.switch_backend('agg')


def load_factor(WIND_yr, year, m, lons, lats,timeS,path):
    '''

    :param WIND_yr: w100 dataset
    :param year: the year, such as 1993
    :param m: the number of model, such as m=1
    :param lons: the longitude 1D
    :param lats: the latitude 1D
    :param timeS: the time series we added, from 0:00
    :param path: the path we save
    :return: no return, just save the .nc dataset of load factor
    '''
    winds = np.array(
        [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5,
         13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5,
         24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
         35])
    # -- 14,Vestas,1074,V136/3450
    power_onshore = np.array(
        [0, 0, 0, 0, 0, 15, 35, 121, 212, 341, 473, 661, 851, 1114, 1377, 1718, 2058, 2456, 2854, 3200, 3415, 3445,
         3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450,
         3450, 3450, 3450, 3450, 3450, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    max_onshore = np.max(power_onshore)
    # --scaling to max value to obtain load factor
    power_onshore = power_onshore / max_onshore
    # --14,Vestas,867,V164/8000
    power_offshore = np.array(
        [0, 0, 0, 0, 0, 0, 0, 40, 100, 370, 650, 895, 1150, 1500, 1850, 2375, 2900, 3525, 4150, 4875, 5600, 6350, 7100,
         7580, 7800, 7920, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000,
         8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0])
    max_offshore = np.max(power_offshore)
    # --scaling to max value to obtain load factor
    power_offshore = power_offshore / max_offshore
    # --find indices where onshore production is not 0 or max value
    ind = np.where((power_onshore != 0) & (power_onshore != max_onshore))  # 找到所有有数值的位置
    deg_onshore = 10  # --degree of polynomial fit
    z_onshore = np.polyfit(winds[ind], power_onshore[ind], deg_onshore)  # 对其进行拟合
    f_onshore = np.poly1d(z_onshore)
    print('fit onshore=', f_onshore)  # 拟合的结果
    # --find indices where onshore production is not 0 or max value
    ind = np.where((power_offshore != 0) & (power_offshore != max_offshore))
    deg_offshore = 10  # --degree of polynomial fit
    z_offshore = np.polyfit(winds[ind], power_offshore[ind], deg_offshore)
    f_offshore = np.poly1d(z_offshore)
    woncfr = np.ones(WIND_yr.shape) * z_onshore[0]
    for i in range(1, deg_onshore + 1):
        print('deg pol onshore=', i, deg_onshore)
        woncfr = woncfr * WIND_yr + z_onshore[i]
    # --putting the 0 and 1
    woncfr[np.where((WIND_yr <= 3.0) | (WIND_yr >= 22.5))] = 0.0
    woncfr[np.where((WIND_yr >= 10.0) & (WIND_yr < 22.5))] = 1.0
    # --OFFSHORE load factor
    wofcfr = np.ones(WIND_yr.shape) * z_offshore[0]
    for i in range(1, deg_offshore + 1):
        print('deg pol offshore=', i, deg_offshore)
        wofcfr = wofcfr * WIND_yr + z_offshore[i]
    # --putting the 0 and 1
    wofcfr[np.where((WIND_yr <= 3.0) | (WIND_yr >= 25.5))] = 0.0
    wofcfr[np.where((WIND_yr >= 12.5) & (WIND_yr < 25.5))] = 1.0
    # open new netcdf dataset in writing mode
    fileout = path
    # fileout='/data/syu/ERA5/10_transfer_100_model/model_apply/test_one_mdel_all/model1/test/load_factor_'+str(year)+'.nc'
    result = nc.Dataset(fileout, 'w')  # , format='NETCDF4_CLASSIC')
    # create dimensions
    lat = result.createDimension('lat', len(lats))
    lat = result.createVariable('lat', np.float32, ('lat',))
    lat.standard_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = "Y"
    lat[:] = lats
    lon = result.createDimension('lon', len(lons))
    lon = result.createVariable('lon', np.float32, ('lon',))
    lon.standard_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = "X"
    lon[:] = lons
    time = result.createDimension('time', None)
    time = result.createVariable('time', np.float64, ('time',))
    time.standard_name = 'time'
    time.long_name = 'time'
    time[:] = timeS
    woncfr_netcdf = result.createVariable('woncfr', np.float64, ('time', 'lat', 'lon'), fill_value=-99999)
    woncfr_netcdf.units = 'kW/kW_installed'
    woncfr_netcdf[:, :, :] = woncfr
    wofcfr_netcdf = result.createVariable('wofcfr', np.float64, ('time', 'lat', 'lon'), fill_value=-99999)
    wofcfr_netcdf.units = 'kW/kW_installed'
    wofcfr_netcdf[:, :, :] = wofcfr
    # global attributes
    result.CDI = "Climate Data Interface version 1.9.5 (http://mpimet.mpg.de/cdi)"
    result.CDO = "Climate Data Operators version 1.9.5 (http://mpimet.mpg.de/cdo)"
    result.NCProperties = "version=1|netcdflibversion=4.6.1|hdf5libversion=1.10.2"
    result.title = "wind power capacity factor"
    result.close()


def write_to_nc(data, timeS, lonS, latS, file_name_path):
    '''
    :param data: 3D dataset we want to save
    :param timeS: time series,1D
    :param lonS: longitude,1D
    :param latS: latitude, 1D
    :param file_name_path: save path
    :return: no return, just save the .nc dataset of load factor
    '''
    da = nc.Dataset(file_name_path, 'w', format='NETCDF4')
    da.createDimension('lon', len(lonS))
    da.createDimension('lat', len(latS))
    da.createDimension('time', len(timeS))
    da.createVariable("lon", 'f', ("lon"))
    da.createVariable("lat", 'f', ("lat"))
    da.createVariable("time", 'f', ("time"))
    da.variables['lat'][:] = latS
    da.variables['lon'][:] = lonS
    da.variables['time'][:] = timeS
    da.createVariable('value', 'f8', ('time', 'lat', 'lon'))
    da.variables['value'][:] = data
    da.close()



if __name__ == "__main__":
   print()
