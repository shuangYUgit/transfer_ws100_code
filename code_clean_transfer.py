#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is used to apply the climate model output (ws10,tas,tas850) on the trained RF model

@author: yushuang
"""
# 1. import packages
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
import function



# 2. Assign a value to a variable
start_year = 1993  # start year
end_year = 1993  # end_year
dd = np.arange(start_year, end_year + 1)
freq_sfcwind = 3  # the frequency of the sfcwind data, 3-hourly
freq_tas = 3  # the frequency of the tas data, 3-hourly
freq_tas850 = 6  # the frequency of the tas_850 data, 6-hourly
freq_save = 6  # the frequency of the save data, 6-hourly



# 3. load the important dataset
framex1 = './data/model_1/sfcWind_1993_remove_bia.nc'
lats = nc.Dataset(framex1)['lat']
lons = nc.Dataset(framex1)['lon']
lon_grid, lat_grid = np.meshgrid(lons, lats)
globe_land_mask = globe.is_land(lat_grid, lon_grid) + 1
lon_new = lon_grid.ravel('F')
lat_new = lat_grid.ravel('F')
rf_model = joblib.load('./RF_80_.joblib')  # load the trained RF model
X_train = np.load('./data/X_train.npy')  # The original dataset which was used to train RF model before
mask = globe.is_land(lat_new, lon_new) + 1  # land-ocean grided, land is 2, ocean is 1
# the information of the model
m = 1  # serial number of the model



# 4. start the application, cycle by the year.
for year in dd:
    # (1) prepare the prediction feature
    sha = nc.Dataset('./data/model_' + str(m) + '/ta850_' + str(year) + '_remove_bia.nc')['value'][
          :].shape  # the shape of the climate model
    timeS = np.array(pd.date_range(str(year) + "-01-01", periods=sha[0],
                                   freq=str(freq_save) + "H"))  # The time series of the climate model
    time_pre = np.arange(0, 8, freq_save / 3)  # prepare for the time feature
    w1 = nc.Dataset('./data/model_' + str(m) + '/sfcWind_' + str(year) + '_remove_bia.nc')['value'][
         :]  # import wind speed dataset
    w_3 = int(
        3 / freq_sfcwind)  # cause the feature is 3-hourly, If the climate model has a higher resolution, determine the multiplier
    w10_pre = np.concatenate(
        (w1[0 * w_3:-8 * w_3, :, :].ravel('F').reshape(-1, 1), w1[3 * w_3:-5 * w_3, :, :].ravel('F').reshape(-1, 1),
         w1[4 * w_3:-4 * w_3, :, :].ravel('F').reshape(-1, 1), w1[5 * w_3:-3 * w_3, :, :].ravel('F').reshape(-1, 1),
         w1[6 * w_3:-2 * w_3, :, :].ravel('F').reshape(-1, 1), w1[7 * w_3:-1 * w_3, :, :].ravel('F').reshape(-1, 1),
         w1[8 * w_3:, :, :].ravel('F').reshape(-1, 1),
         np.repeat(lon_new, w1.shape[0] - 8 * w_3).reshape(-1, 1),
         np.repeat(lat_new, w1.shape[0] - 8 * w_3).reshape(-1, 1),
         np.repeat(mask, w1.shape[0] - 8 * w_3).reshape(-1, 1)),
        axis=1)  # prepare the 10m wind speed feature, longitude, latitude, land-ocean feature
    fw = int(
        freq_save / freq_sfcwind)  # cause the save .nc is 6-hourly, If the climate model has a higher resolution, determine the multiplier
    w10 = w10_pre[::fw, :]  # make the feature of w10 has the same frequency as freq_save
    del w1, w10_pre
    tas_3 = int(3 / freq_tas)
    t2m_pre = nc.Dataset('./data/model_' + str(m) + '/tas_' + str(year) + '_remove_bia.nc')['value'][
              4 * tas_3:-4 * tas_3, :, :].ravel('F').reshape(-1, 1)
    ftas = int(freq_save / freq_tas)
    t2m = t2m_pre[::ftas, :]
    t850_3 = int(6 / freq_tas850)
    t850_pre = nc.Dataset('./data/model_' + str(m) + '/ta850_' + str(year) + '_remove_bia.nc')['value'][
               2 * t850_3:-2 * t850_3, :, :].ravel('F').reshape(-1, 1)
    ft850 = int(freq_save / freq_tas850)
    t850 = t850_pre[::ft850, :]
    t_diff = t2m - t850
    time_pre_apply = np.concatenate((time_pre[int(time_pre.shape[0] / 2):], time_pre[0:int(time_pre.shape[0] / 2)]), 0)
    TIME = np.tile(time_pre_apply, int(t_diff.shape[0] / time_pre_apply.shape[0])).reshape(-1, 1)
    wx_all = np.concatenate((w10, t2m, t850, t_diff, TIME),
                            axis=1)  # feature preparation, including w10-4,w10-1,w10,w10+1,w10+2,w10+3,w10+4,lon,lat,land-ocean
    del t2m, t850, t_diff, TIME, t2m_pre, w10
    loc_nan = np.array(np.where(np.isnan(np.mean(wx_all, axis=1)))).reshape(-1, 1)  # 找到所有有nan的行并且删掉
    w_predictor = np.array(pd.DataFrame(wx_all).dropna(axis=0, how='any'))  # finish the feature
    del wx_all
    # (2) normalization
    sc = StandardScaler()
    X_train2 = sc.fit_transform(X_train)
    X = sc.transform(w_predictor)  # finish the normalization
    del w_predictor
    # (3) apply the trained RF model
    y_t_m = rf_model.predict(X)
    del X, X_train2
    # (4) Processing data format, which has the same shape as the climate model
    a = np.full([sha[1] * sha[2] * (sha[0] - time_pre_apply.shape[0])], np.nan)  # an empty variable，
    h = np.array(range(len(a))).tolist()
    loc = loc_nan.reshape(-1, ).tolist()  # the location we delate in line 79
    yes = np.array(list(set(h) - (set(loc))))  # find the location which we didn't delete
    del loc, h
    for i in range(yes.shape[0]):  # give the value in the location we didn't delet.
        a[yes[i],] = y_t_m[i]
    kb = np.full([sha[0] - time_pre_apply.shape[0], sha[1], sha[2]],
                 np.nan)  # put the dataset in the 3D location we want.
    for i in range(sha[2]):
        for j in range(sha[1]):
            kb[:, j, i] = a[((sha[0] - time_pre_apply.shape[0]) * sha[1] * i + (sha[0] - time_pre_apply.shape[0]) * j):(
                    (sha[0] - time_pre_apply.shape[0]) * sha[1] * i + (sha[0] - time_pre_apply.shape[0]) * (j + 1))]
    k = np.concatenate((np.full([int(time_pre_apply.shape[0] / 2), sha[1], sha[2]], np.nan), kb,
                        np.full([int(time_pre_apply.shape[0] / 2), sha[1], sha[2]], np.nan)),
                       axis=0)  # Give the beginning and the end an empty set， make the shape is same with the climate model
    # （5）save the dataset, ws100 and load factor
    function.write_to_nc(k, timeS, lons, lats, './result/model_' + str(m) + '/ws100/predict_' + str(year) + '_ws100.nc')
    path = './result/model_' + str(m) + '/load_factor/load_factor_' + str(year) + '.nc'
    function.load_factor(k, year, m, lons, lats, timeS, path)
    del k, a, loc_nan, yes, kb
