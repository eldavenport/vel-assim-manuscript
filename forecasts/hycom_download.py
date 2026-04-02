import xarray as xr
import requests
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

def download_hycom_2013_fast(output_dir="../data/hycom_reanalysis_2013"):
    """
    Download HYCOM 2013 data with aggressive chunking for speed
    Domain: 180W-90W, 10S-10N, 0-500m depth
    Variables: T, S, U, V, SSH
    """
    
    url = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2013"
    
    variables = {
        'temperature': 'water_temp',
        'salinity': 'salinity',
        'u_velocity': 'water_u', 
        'v_velocity': 'water_v',
        'ssh': 'surf_el'
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Using aggressive chunking for faster OPeNDAP downloads...")
    
    # Open dataset once
    print("Opening HYCOM dataset...")
    ds = xr.open_dataset(url, decode_times=False, engine='netcdf4')
    
    # Select spatial domain
    domain = ds.sel(
        lat=slice(-10, 10),
        lon=slice(-180, -90),
        depth=slice(0, 500)  
    )
    
    print(f"Domain shape: {dict(domain.sizes)}")
    total_timesteps = domain.sizes['time']
    
    # Download each variable in 2 large chunks (6 months each)
    for var_name, hycom_var in variables.items():
        print(f"\nDownloading {var_name}...")
        
        var_data = domain[hycom_var]
        
        # SSH doesn't have depth 
        if var_name == 'ssh':
            var_data = var_data.isel(depth=0)
        
        # Split into 12 monthly chunks - more reasonable for server
        chunk_size = total_timesteps // 12
        
        for month in range(1, 13):
            start_idx = (month-1) * chunk_size
            end_idx = month * chunk_size if month < 12 else total_timesteps
            period = f"{month:02d}"
                
            print(f"  {period}: downloading timesteps {start_idx}-{end_idx}...")
            
            start_time = time.time()
            
            try:
                # Download monthly chunk - should work better than huge 6-month chunks
                monthly_data = var_data.isel(time=slice(start_idx, end_idx)).load()
                
                download_time = time.time() - start_time
                size_mb = monthly_data.nbytes / (1024**2)
                speed = size_mb / download_time if download_time > 0 else 0
                
                print(f"    Downloaded {size_mb:.1f} MB in {download_time:.1f}s ({speed:.1f} MB/s)")
                
                # Save monthly file
                output_file = output_path / f"{var_name}_2013_{period}.nc"
                monthly_data.to_netcdf(output_file)
                
                print(f"    ✓ Saved {output_file.name}")
                
            except Exception as e:
                print(f"    ✗ Failed to download {var_name} month {period}: {e}")
                continue
    
    ds.close()
    print(f"\n✓ Fast download completed! Files saved to {output_path}")
    return output_path

def download_hycom_2013_ncss(output_dir="../data/hycom_reanalysis_2013"):
    """
    Download HYCOM 2013 data using NCSS for fast downloads
    Domain: 180W-90W, 10S-10N, 0-500m depth
    Variables: T, S, U, V, SSH
    """
    
    # NCSS endpoint - much faster than OPeNDAP
    ncss_base = "https://ncss.hycom.org/thredds/ncss/grid/GLBv0.08/expt_53.X/data/2013"
    
    variables = {
        'temperature': 'water_temp',
        'salinity': 'salinity',
        'u_velocity': 'water_u', 
        'v_velocity': 'water_v',
        'ssh': 'surf_el'
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Using NCSS for fast downloads...")
    
    # Download each variable by quarter (3 months) for speed
    quarters = [
        ("Q1", "2013-01-01T00:00:00Z", "2013-03-31T23:59:59Z"),
        ("Q2", "2013-04-01T00:00:00Z", "2013-06-30T23:59:59Z"), 
        ("Q3", "2013-07-01T00:00:00Z", "2013-09-30T23:59:59Z"),
        ("Q4", "2013-10-01T00:00:00Z", "2013-12-31T23:59:59Z")
    ]
    
    for var_name, hycom_var in variables.items():
        print(f"\nDownloading {var_name}...")
        
        for quarter_name, start_time, end_time in quarters:
            print(f"  {quarter_name}: {start_time[:10]} to {end_time[:10]}...")
            
            # NCSS request parameters
            params = {
                'var': hycom_var,
                'north': 10.0,
                'south': -10.0,
                'west': -180.0,
                'east': -90.0,
                'time_start': start_time,
                'time_end': end_time,
                'accept': 'netcdf4'
            }
            
            # Add vertical subset for 3D variables
            if var_name != 'ssh':
                params['vertCoord'] = '0:500'
            
            start_time_dl = time.time()
            
            try:
                # Make NCSS request
                print(f"    Requesting {var_name} data...")
                response = requests.get(ncss_base, params=params, timeout=300)
                response.raise_for_status()
                
                download_time = time.time() - start_time_dl
                size_mb = len(response.content) / (1024**2)
                speed = size_mb / download_time if download_time > 0 else 0
                
                print(f"    Downloaded {size_mb:.1f} MB in {download_time:.1f}s ({speed:.1f} MB/s)")
                
                # Save quarterly file
                output_file = output_path / f"{var_name}_2013_{quarter_name}.nc"
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"    ✓ Saved {output_file.name}")
                
            except Exception as e:
                print(f"    ✗ Error downloading {var_name} {quarter_name}: {e}")
                continue
    
    print(f"\n✓ Fast NCSS download completed! Files saved to {output_path}")
    return output_path

def download_hycom_2013_test(output_dir="../data/hycom_reanalysis_2013"):
    """
    Test function - downloads just one small chunk of HYCOM data
    """
    url = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/2013"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Opening HYCOM dataset...")
    ds = xr.open_dataset(url, decode_times=False, engine='netcdf4')
    
    # Get very small subset: 1 time step, small spatial area
    print("Selecting small subset...")
    test_data = ds.water_temp.isel(
        time=0,                 # Just first time step
        depth=slice(0, 5),      # Just first 5 depths  
        lat=slice(1500, 1600),  # Small lat range
        lon=slice(2000, 2100)   # Small lon range
    )
    
    print(f"Test data shape: {test_data.shape}")
    print("Loading data...")
    
    # Load the data
    loaded_data = test_data.load()
    
    # Save test file
    test_file = output_path / "test_temperature.nc"
    print(f"Saving to {test_file}...")
    loaded_data.to_netcdf(test_file)
    
    ds.close()
    print("✓ Test completed successfully!")
    return test_file