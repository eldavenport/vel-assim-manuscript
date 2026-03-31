import xarray as xr
from xmitgcm import open_mdsdataset
import numpy as np
import warnings
# warnings.filterwarnings("ignore")

def tpose2012to2013(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 31+29 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*num_diags,itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'
    ref_date = '2011-12-31 12:0:0'
    delta_t = 1440/itPerFile * 60
    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_date,delta_t=delta_t)

    folder_months = ['mar2012/diags/','may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/','nov2012/diags_new/']
    ref_dates = ['2012-02-29 12:0:0', '2012-04-30 12:0:0', '2012-06-30 12:0:0', '2012-08-31 12:0:0', '2012-10-31 12:0:0']
    folder_days = np.array([61,61,62,61,61])
    itPerFile = [72, 72, 72, 48, 72]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        delta_t = 1440/itPerFile[i] * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds

        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_dates[i],delta_t=delta_t)
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    #2013-2014
    data_parent_dir = '/data/SO3/averdy/TPOSE6/'

    num_diags = 31+28 + offset #
    itPerFile = 72 # 1 day
    intervals = range(offset*itPerFile,itPerFile*num_diags,itPerFile)
    data_dir = data_parent_dir + 'jan2013/diags_daily/'
    ref_date = '2012-12-31 12:0:0'
    delta_t = 1440/itPerFile * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds
    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_date,delta_t=delta_t)
    tpose_ds = xr.concat([tpose_ds,new],'time')

    folder_months = ['mar2013/diags_daily/','may2013/diags_daily/','jul2013/diags_daily/','sep2013/diags_daily/','nov2013/diags_daily/']
    ref_dates = ['2013-02-28 12:0:0', '2013-04-30 12:0:0', '2013-06-30 12:0:0', '2013-08-31 12:0:0', '2013-10-31 12:0:0']
    folder_days = np.array([61,61,62,61,52])
    itPerFile = np.array([72,72,72,72,72])

    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        delta_t = 1440/itPerFile[i] * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds
        
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_dates[i],delta_t=delta_t)
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    print('Days in 2012-2013: (should be 731)')
    print(len(tpose_ds.time))

    # tpose_ds = tpose_ds.chunk({
    # 'XC': 1,   # horizontal cell-centered
    # 'XG': 1,   # horizontal staggered
    # 'YC': -1,    # single chunk along Y to speed up mean reductions
    # 'YG': -1,    # same for staggered Y
    # 'Z': -1,      # vertical levels
    # 'time': -1
    # })
    return tpose_ds

def tpose2012to2013_kpp(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 61 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*num_diags,itPerFile)
    data_dir = data_parent_dir + 'mar2012/diags/'
    ref_date = '2012-03-01 0:0:0'
    delta_t = 1440/itPerFile * 60
    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_date,delta_t=delta_t)

    folder_months = ['may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/']
    ref_dates = ['2012-05-01 0:0:0', '2012-07-01 0:0:0', '2012-09-01 0:0:0']
    folder_days = np.array([61,62,112])
    itPerFile = [72, 72, 48]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        delta_t = 1440/itPerFile[i] * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds

        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_dates[i],delta_t=delta_t)
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    #2013-2014
    data_parent_dir = '/data/SO3/averdy/TPOSE6/'

    num_diags = 31+28 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*num_diags,itPerFile)
    data_dir = data_parent_dir + 'jan2013/diags_daily/'
    ref_date = '2012-12-31 0:0:0'
    delta_t = 1440/itPerFile * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds
    new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_date,delta_t=delta_t)
    tpose_ds = xr.concat([tpose_ds,new],'time')

    folder_months = ['mar2013/diags_daily/','may2013/diags_daily/','jul2013/diags_daily/','sep2013/diags_daily/','nov2013/diags_daily/']
    ref_dates = ['2013-02-28 0:0:0', '2013-04-30 0:0:0', '2013-06-30 0:0:0', '2013-08-31 0:0:0', '2013-10-31 0:0:0']
    folder_days = np.array([61,61,62,61,52])
    itPerFile = np.array([72,72,72,72,72])

    i = 0
    for month in folder_months:
        print(month)
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        delta_t = 1440/itPerFile[i] * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds
        
        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_dates[i],delta_t=delta_t)
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    print('Days in 2012-2013: (should be 731)')
    print(len(tpose_ds.time))
    return tpose_ds


def tpose2012(prefix):

    print('opening 2012')
    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    offset = 10 # number of days to offset the start of each run (in order to avoid artifacts)

    num_diags = 31+29 + offset #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*num_diags,itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'
    ref_date = '2011-12-31 12:0:0'
    delta_t = 1440/itPerFile * 60 
    tpose_ds = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_date,delta_t=delta_t)

    folder_months = ['mar2012/diags/','may2012/diags/','jul2012/diags/','sep2012/diags_iter7_daily/','nov2012/diags_new/']
    ref_dates = ['2012-02-29 12:0:0', '2012-04-30 12:0:0', '2012-06-30 12:0:0', '2012-08-31 12:0:0', '2012-10-31 12:0:0']
    folder_days = np.array([61,61,62,61,52]) # december should capture the last day of the year (need to be end inclusive)
    itPerFile = [72, 72, 72, 48,72]
    i = 0

    for month in folder_months:
        num_diags = folder_days[i] + offset
        intervals = range(offset*itPerFile[i],itPerFile[i]*num_diags,itPerFile[i])
        data_dir = data_parent_dir + month
        delta_t = 1440/itPerFile[i] * 60  # calculate the minutes per timestep and multiply by 60 to get delta_t in seconds

        new = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix,ref_date=ref_dates[i],delta_t=delta_t)
        tpose_ds = xr.concat([tpose_ds,new],'time')
        i += 1

    # 336//4 = 84 (latitude)
    # could change this to 'time':len(ds.time)//4, etc
    # tpose_ds.chunk({'time':1,'XG':141,'XC':141,'YC':84,'YG':84,'Z':11,'Zl':11})
    print('Days in 2012: (should be 366)')
    print(len(tpose_ds.time))
    return tpose_ds


def tpose2012_4month(prefix):

    #2012
    data_parent_dir = '/data/SO6/TPOSE_diags/tpose6/'
    grid_dir = '/data/SO6/TPOSE_diags/tpose6/grid_6/'

    num_diags = 31+29+30+30  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jan2012/diags/'

    dsJan = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix)

    num_diags = 122  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'mar2012/diags/'

    dsMar = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix)

    num_diags = 123  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'may2012/diags/'

    dsMay = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix)

    num_diags = 123  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'jul2012/diags/'

    dsJul = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix)

    num_diags = 121  #
    itPerFile = 48 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'sep2012/diags_iter7_daily/'

    dsSep = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix)

    num_diags = 120  #
    itPerFile = 72 # 1 day
    intervals = range(itPerFile,itPerFile*(num_diags+1),itPerFile)
    data_dir = data_parent_dir + 'nov2012/diags_new/'

    dsNov = open_mdsdataset(data_dir=data_dir,grid_dir=grid_dir,iters=intervals,prefix=prefix)




    return dsJan, dsMar, dsMay, dsJul, dsSep, dsNov