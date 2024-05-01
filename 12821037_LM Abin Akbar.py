#Nama : LM Abin Akbar
#NIM  : 12821037
# Tugas Time-Lag Esembles

import warnings
warnings.simplefilter('ignore') #ignores simple warning

import numpy as np
from datetime import datetime

from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#catalog
cat=TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231017_0000.grib2/catalog.xml')

#dataset
ds=list(cat.datasets.values())[0]
#subset
ncss=ds.subset()
#query
query=ncss.query()

dir(query)

ncss.variables

#dir(query)
tstart=datetime(2023,10,24,0,0)
tend=datetime(2023,10,26,21,0)

"""### **Precipitation_rate_surface**"""

varname='Precipitation_rate_surface'
query.lonlat_box(north=15, south=-15, east=150, west=95)
query.time_range(start=tstart,end=tend)
query.accept('netcdf4')
query.variables(varname)

#get data
data = ncss.get_data(query)
data = xr.open_dataset(NetCDF4DataStore(data))
data

import warnings
warnings.simplefilter('ignore') #ignores simple warning
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore
import xarray as xr

def timelagGFS(file_list,varname,domain,valid_time_start,valid_time_end):
  '''
  Fungsi sederhana untuk membuat time-lag ensemble dari data GFS
  '''
  #member-1
  m= TDSCatalog(file_list[0])
  print(m)
  ds=list(m.datasets.values())[0]
  ncss=ds.subset()
  query=ncss.query()
  query.lonlat_box(north=domain['north'], south=domain['south'], east=domain['east'], west=domain['west'])
  query.time_range(start=valid_time_start,end=valid_time_end)
  query.accept('netcdf4')
  query.variables(varname)
  #get data
  data = xr.open_dataset(NetCDF4DataStore(ncss.get_data(query)))
  data=data.rename({data[varname].dims[0]: 'time'})

  #member lainnya
  for mem in file_list[1:]:
    m= TDSCatalog(mem)
    print(m)
    ds=list(m.datasets.values())[0]
    ncss=ds.subset()

    query=ncss.query()
    query.lonlat_box(north=llbox['north'], south=llbox['south'], east=llbox['east'], west=llbox['west'])
    query.time_range(start=valid_s,end=valid_e)
    query.accept('netcdf4')
    query.variables(varname)

    #get data
    dat = xr.open_dataset(NetCDF4DataStore(ncss.get_data(query)))
    dat=dat.rename({dat[varname].dims[0]: 'time'})

    #concatenate
    data=xr.concat([data,dat],"member")

  return data

time_lag_list=['https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231023_0000.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231023_0600.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231023_1200.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231023_1800.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231022_0000.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231022_0600.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231022_1200.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231022_1800.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231021_0000.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231021_0600.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231021_1200.grib2/catalog.xml',
               'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_20231021_1800.grib2/catalog.xml']

#parameter subset
varname='Precipitation_rate_surface'
llbox={'north':15,
       'south':-15,
       'west':90,
       'east':150}
valid_s=datetime(2023,10,24,0,0)
valid_e=datetime(2023,10,26,21,0)

tlag_ens=timelagGFS(time_lag_list,varname,llbox,valid_s,valid_e)

tlag_ens[varname]

tlag_ens_3=tlag_ens*3600*3 #akumulasi ch 3 jam (mm/3jam)

time='2023-10-24 00:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 03:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 06:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 09:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 12:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 15:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 18:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 21:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 00:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 03:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 06:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 09:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 12:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 15:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 18:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 21:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 00:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 03:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 06:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 09:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 12:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 15:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 18:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 21:00'

thold=0.7 #threshold

#probabilitas value > thold
dat=tlag_ens_3[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/12

#gambar
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.4,
          vmax=1.0,
          cmap='Blues',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

"""### **Total_cloud_cover_entire_atmosphere**"""

varname='Total_cloud_cover_entire_atmosphere'
cl_cover=timelagGFS(time_lag_list,varname,llbox,valid_s,valid_e)

time='2023-10-24 00:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 03:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 06:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 09:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 12:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 15:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 18:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 21:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 00:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 03:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 06:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 09:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 12:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 15:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 18:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 21:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 00:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 03:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 06:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 09:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 12:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 15:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 18:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 21:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='spring',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

"""###**Vegetation_surface**"""

varname='Vegetation_surface'
cl_cover1=timelagGFS(time_lag_list,varname,llbox,valid_s,valid_e)

time='2023-10-24 00:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 03:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 06:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 09:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 12:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 15:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 18:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-24 21:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 00:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 03:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 06:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 09:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 12:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 15:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 18:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-25 21:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 00:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 03:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 06:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 09:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 12:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 15:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 18:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

time='2023-10-26 21:00'
thold=70.0
#probabilitas value > thold
dat=cl_cover1[varname].sel(time=time)
prob=np.sum(dat>thold,axis=0)/8
#
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([95, 150, -10, 10])
ax.add_feature(cfeature.BORDERS, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.gridlines(draw_labels=True,linestyle=':')

prob.plot(transform=ccrs.PlateCarree(),
          vmin=0.0,
          vmax=1.0,
          cmap='BuPu',
          cbar_kwargs={'shrink': 0.4,
                       'label':'Probability'},
          )

"""### **Meteogram**

Untuk Precipitation_rate_surface
"""

rata2 = tlag_ens_3.resample(time='3H').mean()

import matplotlib.pyplot as plt
import seaborn
import pandas as pd

ts = pd.Series(rata2['Precipitation_rate_surface'].sel(longitude=107.609810,
                                                            latitude=-6.914744,
                                                            method='nearest').mean(dim='member').values,
               index = rata2['time'].values)

fig, ax = plt.subplots(figsize=(20,5))
seaborn.boxplot(x = ts.index.dayofyear,
                y = ts,
                ax = ax)

plt.title("Meteogram untuk Precipitation rate surface")
plt.ylabel("Precipitation", fontsize='12', fontweight='bold')
plt.xlabel("Tanggal", fontsize='12', fontweight='bold')

"""Untuk Vegetation_surface"""

rata_rata = cl_cover1.resample(time='3H').mean()

import matplotlib.pyplot as plt
import seaborn
import pandas as pd

ts = pd.Series(rata_rata['Vegetation_surface'].sel(longitude=107.609810,
                                                            latitude=-6.914744,
                                                            method='nearest').mean(dim='member').values,
               index = rata_rata['time'].values)

fig, ax = plt.subplots(figsize=(20,5))
seaborn.boxplot(x = ts.index.dayofyear,
                y = ts,
                ax = ax)

plt.title("Meteogram untuk Vegetation surface")

"""Untuk Total_cloud_cover_entire_atmosphere"""

rata_rata_k = cl_cover.resample(time='3H').mean()

import matplotlib.pyplot as plt
import seaborn
import pandas as pd

ts = pd.Series(rata_rata_k['Total_cloud_cover_entire_atmosphere'].sel(longitude=107.609810,
                                                            latitude=-6.914744,
                                                            method='nearest').mean(dim='member').values,
               index = rata_rata_k['time'].values)

fig, ax = plt.subplots(figsize=(20,5))
seaborn.boxplot(x = ts.index.dayofyear,
                y = ts,
                ax = ax)

plt.title("Meteogram untuk Total cloud cover entire atmosphere")