import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
import datetime
from collections import defaultdict

synoptic_source_interp = None
initial_lats = None
initial_profile = None

def get_initial_profile_from_wso(lat_points, B_unit, data_dir="data/24",
                                 unit_to_gauss=1.0):
    """
    unit_to_gauss : multiply raw WSO values by this before normalizing.
        WSO synoptic maps are in microtesla; use 0.01 to work in true Gauss,
        or 1.0 (default) to keep WSO native units.
    """
    days_since_start, lats_deg_src, syn_map_src, _ = build_synoptic_map(data_dir)

    lat_sort_idx = np.argsort(lats_deg_src)
    lats_deg_src = np.asarray(lats_deg_src)[lat_sort_idx]
    syn_map_src = np.asarray(syn_map_src)[:, lat_sort_idx]

    model_lats_deg = np.linspace(-90.0, 90.0, int(lat_points))

    f = interp1d(lats_deg_src, syn_map_src[0], kind="cubic",
                 bounds_error=False, fill_value="extrapolate")
    init_profile = f(model_lats_deg) * unit_to_gauss / B_unit   # model units
    lat_rad = np.deg2rad(model_lats_deg)
    w = np.cos(lat_rad)
    w = np.clip(w,0.0,None)
    w/=w.sum()
    init_profile = init_profile - np.sum(init_profile*w)
    return model_lats_deg, init_profile
    
def load_full_wso_data(filepath):
    data = []
    ct_labels = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("CT"):
            ct_label = lines[i].split()[0]  # e.g., CT2293:360
            ct_labels.append(ct_label)

            row_data = []

            for j in range(4):  # Read 4 lines per synoptic row
                line = lines[i + j].strip().split()

                # Skip the CT label on the first line
                if j == 0 and ':' in line[0]:
                    line = line[1:]

                # Convert remaining strings to float
                row_data.extend(map(float, line))

            if len(row_data) != 30:
                raise ValueError(f"Expected 30 values per CT row, got {len(row_data)} at {ct_label}")

            data.append(row_data)
            i += 4  # Move to next CT block
        else:
            i += 1

    return np.array(data), ct_labels


def build_synoptic_map(data_dir="data/24", lat_points=360):
    file_paths = sorted(glob.glob(os.path.join(data_dir, "WSO.*.F.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No WSO files found in {data_dir}")
    full_map = []
    rotation_nums = []
    data_rows=[]

    # WSO latitudes
    sinlats = np.linspace(14.5 / 15, -14.5 / 15, 30)
    lats_deg = np.arcsin(sinlats) * 180 / np.pi
    model_lats = np.linspace(-90, 90, lat_points)

    # Reference Carrington rotation number and date
    if data_dir =="data/25":
        ref_rot = 2225  
        ref_date = datetime.datetime(2019, 12, 10)  
    if data_dir =="data/24":
        ref_rot = 2078  
        ref_date = datetime.datetime(2008, 12, 17) 
    if data_dir =="data/23":
        ref_rot = 1913  
        ref_date = datetime.datetime(1996, 8, 22)  
    if data_dir =="data/22":
        ref_rot = 1780  
        ref_date = datetime.datetime(1986, 9, 16)  
    if data_dir =="data/21":
        ref_rot = 1614 
        ref_date = datetime.datetime(1974, 4, 24)  
    print (ref_date)
    carrington_period_days = 27.2753

    for filepath in file_paths:
        with open(filepath, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].startswith("CT"):
                ct_label = lines[i].split()[0]  # e.g., CT2293:360
                rot_num = int(ct_label[2:6])
                longitude = int(ct_label.split(":")[1])
                #print(rot_num)
                row_data = []
                for j in range(4):
                    parts = lines[i + j].strip().split()
                    if j == 0 and ':' in parts[0]:
                        parts = parts[1:]
                    row_data.extend(map(float, parts))
                i += 4

                if len(row_data) == 30:
                    data_rows.append({
                        'rotation':rot_num,
                        'longitude':longitude,
                        'flux':np.array(row_data)
                        })
                    # Interpolate to model lat grid
                    #interp_func = interp1d(lats_deg, row_data, kind='cubic', fill_value='extrapolate')
                    #full_map.append(interp_func(model_lats))
                    rotation_nums.append(rot_num)
            else:
                i += 1
    # Group flux arrays by rotation number
    rotation_flux_map = defaultdict(list)
    for entry in data_rows:
        rot = entry['rotation']
        rotation_flux_map[rot].append(entry['flux'])
    rotation_nums = sorted(rotation_flux_map.keys())
    mean_profiles = []
    for rot in rotation_nums:
        fluxes = np.array(rotation_flux_map[rot])  # shape: (N_longitudes, 30)
        mean_profile = np.mean(fluxes, axis=0)     # average over longitude
        mean_profiles.append(mean_profile)
    #print (len(mean_profiles))

    # Convert rotation numbers to datetime
    dates = [ref_date + datetime.timedelta(days=(rot - ref_rot) * 27.2753) for rot in rotation_nums]
    # Convert datetime to float years (for plotting)
    years_float = np.array([d.year + d.timetuple().tm_yday / 365.25 for d in dates])
    days_since_start = np.array([(d - ref_date).days for d in dates])

    synoptic_map = np.array(mean_profiles)  # shape: (num_rotations, lat_points)
    return days_since_start, lats_deg, synoptic_map, dates


def _remove_monopole_per_time(B_2D, lat_deg):
    lat_rad = np.deg2rad(lat_deg)
    w = np.cos(lat_rad)
    w = np.clip(w, 0.0, None)
    w /= w.sum()
    means = B_2D @ w[:, None]
    return B_2D - means

def _build_interp_from_arrays(t_vec, lat_deg, B_2D):
    T, L = np.meshgrid(t_vec, lat_deg, indexing="ij")
    pts  = np.column_stack([T.ravel(), L.ravel()])
    vals = B_2D.ravel()
    return LinearNDInterpolator(pts, vals, fill_value=0.0)

def get_wso_constraints(Tmax, lat_points, time_steps, B_unit, data_dir="data/24",
                        unit_to_gauss=1.0, max_abs_lat_deg=75.0):
    """
    Build supervised PINN constraints (obs_X, obs_Y) from WSO maps using a
    SINGLE, GLOBALLY-NORMALIZED time axis across the entire directory.

    Args
    ----
    Tmax : float
        Normalized simulation end time (usually 1.0).
    lat_points : int
        Number of latitude points on the model grid.
    time_steps : int
        (Optional) Desired number of time samples; if < total available, we subsample.
    B_unit : float
        Field normalization (divide by this factor).
    data_dir : str
        Folder containing WSO.*.F.txt files.
    unit_to_gauss : float
        Multiply raw WSO values (microtesla) by this. 0.01 -> true Gauss,
        1.0 (default) -> keep WSO native units.
    max_abs_lat_deg : float
        Exclude observation points with |lat| above this. WSO only measures
        up to |lat| ~ 75.2 deg; poleward values are cubic-extrapolation
        artifacts and must NOT be fitted -- the physics fills the polar caps.

    Returns
    -------
    obs_X : (N, 2) array  with columns [lam_norm, t_norm]
    obs_Y : (N, 1) array  target field values in model units
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # 1) Build a global synoptic map and global time axis (DAYS since a ref date)
    #    This function already averages over Carrington longitudes per rotation.
    days_since_start, lats_deg_src, syn_map_src, _ = build_synoptic_map(data_dir)

    # 2) Sort latitudes ascending and reorder the map accordingly (safety)
    lat_sort_idx = np.argsort(lats_deg_src)
    lats_deg_src = np.asarray(lats_deg_src)[lat_sort_idx]
    syn_map_src  = np.asarray(syn_map_src)[:, lat_sort_idx]   # (Nt, Nlat_src)

    # 3) Interpolate each time slice to the model latitude grid
    model_lats_deg = np.linspace(-90.0, 90.0, int(lat_points))
    syn_map_model = np.empty((len(days_since_start), len(model_lats_deg)), dtype=float)
    for k, row in enumerate(syn_map_src):
        f = interp1d(lats_deg_src, row, kind="cubic", bounds_error=False, fill_value="extrapolate")
        syn_map_model[k, :] = f(model_lats_deg)
    syn_map_model = syn_map_model * unit_to_gauss
    syn_map_model = _remove_monopole_per_time(syn_map_model,model_lats_deg)
    # 4) Convert GLOBAL days → GLOBAL years → GLOBAL normalized time in [0, Tmax]
    t_years = np.asarray(days_since_start, dtype=float) / 365.25
    t_norm  = (t_years - t_years.min()) / (t_years.max() - t_years.min()) * Tmax

    # Optional: subsample time uniformly to 'time_steps' if requested
    # Resample the observed map in time to exactly `time_steps` samples
    if time_steps is not None and time_steps > 0:
        t_target = np.linspace(t_norm.min(), t_norm.max(), time_steps)
        syn_map_resampled = np.empty((time_steps, syn_map_model.shape[1]), dtype=float)
        for j in range(syn_map_model.shape[1]):
            f_t = interp1d(t_norm, syn_map_model[:, j], kind="linear",
                           bounds_error=False, fill_value="extrapolate")
            syn_map_resampled[:, j] = f_t(t_target)
        t_norm_sub = t_target
        syn_map_model = syn_map_resampled
    else:
        t_norm_sub = t_norm

    # 5) Build (obs_X, obs_Y) keeping only latitudes actually observed by WSO.
    #    FIX: the mask used to be all-ones, so cubic-extrapolated polar values
    #    (and points at lam = +-0.5, outside the +-0.495 geometry) were fitted
    #    with the highest loss weight. Restrict to |lat| <= max_abs_lat_deg.
    obs_X = []
    obs_Y = []
    mask_lat = np.abs(model_lats_deg) <= float(max_abs_lat_deg)

    # Normalize latitude to lam_norm ≈ lat/180 ∈ [-0.5, 0.5]
    lam_norm_all = model_lats_deg / 180.0

    for ti, tn in enumerate(t_norm_sub):
        row = syn_map_model[ti, :]
        for j, use in enumerate(mask_lat):
            if not use:
                continue
            obs_X.append([lam_norm_all[j], tn])
            obs_Y.append(row[j] / B_unit)

    obs_X = np.asarray(obs_X, dtype=float)
    obs_Y = np.asarray(obs_Y, dtype=float).reshape(-1, 1)

    print(f"[get_wso_constraints] {data_dir} → Nt={len(t_norm)} "
          f"(used {len(t_norm_sub)}), Nlat={lat_points}, total points={len(obs_X)}")
    return obs_X, obs_Y


def get_wso_map_for_comparison(Tmax, lat_points, time_steps, B_unit, data_dir="data/24",
                               unit_to_gauss=1.0):
    """
    Return the observed WSO synoptic map on the same normalized time axis and
    model latitude grid used by the PINN, in model units.

    unit_to_gauss : multiply raw WSO values (microtesla) by this.
        0.01 -> true Gauss, 1.0 (default) -> WSO native units.

    Returns
    -------
    t_norm_sub : (Nt,) array
        Normalized time in [0, Tmax]
    model_lats_deg : (Nlat,) array
        Uniform latitude grid in degrees
    B_obs_model : (Nt, Nlat) array
        Observed field in model units
    """
    days_since_start, lats_deg_src, syn_map_src, _ = build_synoptic_map(data_dir)

    # sort latitude
    lat_sort_idx = np.argsort(lats_deg_src)
    lats_deg_src = np.asarray(lats_deg_src)[lat_sort_idx]
    syn_map_src = np.asarray(syn_map_src)[:, lat_sort_idx]

    # interpolate to model grid
    model_lats_deg = np.linspace(-90.0, 90.0, int(lat_points))
    syn_map_model = np.empty((len(days_since_start), len(model_lats_deg)), dtype=float)

    for k, row in enumerate(syn_map_src):
        f = interp1d(lats_deg_src, row, kind="cubic",
                     bounds_error=False, fill_value="extrapolate")
        syn_map_model[k, :] = f(model_lats_deg)
    syn_map_model = syn_map_model * unit_to_gauss
    syn_map_model = _remove_monopole_per_time(syn_map_model, model_lats_deg)
    # global normalized time
    t_years = np.asarray(days_since_start, dtype=float) / 365.25
    t_norm = (t_years - t_years.min()) / (t_years.max() - t_years.min()) * Tmax

    # optional subsampling
    if time_steps is not None and time_steps > 0:
        t_target = np.linspace(t_norm.min(), t_norm.max(), time_steps)
        syn_map_resampled = np.empty((time_steps, syn_map_model.shape[1]), dtype=float)
        for j in range(syn_map_model.shape[1]):
            f_t = interp1d(t_norm, syn_map_model[:, j], kind="linear",
                           bounds_error=False, fill_value="extrapolate")
            syn_map_resampled[:, j] = f_t(t_target)
        t_norm_sub = t_target
        syn_map_model = syn_map_resampled
    else:
        t_norm_sub = t_norm

    # return in model units
    B_obs_model = syn_map_model / B_unit

    return t_norm_sub, model_lats_deg, B_obs_model
# === INITIAL PROFILE SETUP ===



def build_initial_profile(B_unit, data_dir="data/24"):
    file_paths = sorted(glob.glob(os.path.join(data_dir, "WSO.*.F.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No WSO files found in {data_dir}")
    #print(file_paths[0])
    from src.extract import load_full_wso_data  # Safe even within same file
    data_rows, ct_rows = load_full_wso_data(file_paths[0])  # Just the first map

    sinlats = np.linspace(14.5 / 15, -14.5 / 15, 30)
    lats_deg = np.arcsin(sinlats) * 180 / np.pi
    model_lats_deg = np.linspace(-90, 90, 360)  # match model grid

    f_interp = interp1d(lats_deg, data_rows[0], kind='cubic', bounds_error=False, fill_value='extrapolate')
    profile = f_interp(model_lats_deg) / B_unit  # normalize

    return profile

# Set the variable here
#initial_profile = build_initial_profile(B_unit=10.0, data_dir="data/24")



def plot_synoptic_map(data_dir="data/24", lat_points=360, save_path=None):
    # Get time, latitude, synoptic map, and dates
    days_since_start, lats_deg, syn_map, dates = build_synoptic_map(data_dir, lat_points)

    # Sort latitudes to be ascending for plotting
    lat_sort_idx = np.argsort(lats_deg)
    lats_deg_sorted = lats_deg[lat_sort_idx]
    syn_map_sorted = syn_map[:, lat_sort_idx]

    # Convert days to fractional years for smoother plotting
    years_float = np.array([d.year + d.timetuple().tm_yday / 365.25 for d in dates])

    plt.figure(figsize=(12, 6))
    plt.contourf(years_float, lats_deg_sorted, syn_map_sorted.T, levels=100, cmap="RdBu_r")
    cycle_label = os.path.basename(data_dir)  # e.g., "23" from "data/23"
    plt.title(f"WSO Synoptic Map - Solar Cycle {cycle_label}")
    plt.xlabel("Year")
    plt.ylabel("Latitude [°]")
    plt.colorbar(label="Magnetic Field [G]")

    # Set yearly ticks on x-axis
    plt.xticks(
        np.arange(int(years_float[0]), int(years_float[-1]) + 1, 1),
        rotation=45
    )

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Synoptic map saved to {save_path}")
