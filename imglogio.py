"""
Image log input/output functions.
"""

import sys, math
import pandas as pd
import numpy as np
from skimage.transform import resize
from scipy.stats import circmean
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from typing import List, Dict, Tuple, Optional, Sequence
from PIL import Image


def read_csv(fpath: str, 
             azimuth_col: bool = False, 
             vlim: tuple = (None, None)):
	"""
	Read image log data from a CSV file.

	Args:
		fpath (str): File path.
		azimuth_col (bool): Whether the column name contains azimuth.
							Defaults to False.
		vlim (tuple): Clip log values, in the format of (min, max).
					  Defaults to (None, None), which is not to clip the log values.
		
	Returns:
		value (numpy.ndarray): Log value.
		z (numpy.ndarray): Measured depth.
		phi (numpy.ndarray): Azimuth.
		units (dict): {'z': Measured depth unit, 
					   'value': Logging value unit}.
	"""
	# Load data.
	df = pd.read_csv(fpath, skipinitialspace=True, dtype=object)

	# Get azimuthal angles.
	if azimuth_col:
		phi = df.columns[1:]
		phi = np.array(phi, dtype=np.float64)
	else:
		n_phi = len(df.columns) - 1
		phi = np.linspace(0, 360, n_phi, endpoint=False)

	# Get tool depths.
	z = df.values[1:, 0].astype(np.float64)

	# Get log values.
	value = df.values[1:, 1:].astype(np.float64)

	# Clip log values if needed.
	vmin, vmax = vlim
	if vmin != None or vmax != None:
		value = np.clip(value, a_min=vmin, a_max=vmax)

	# Get units.
	units = {}
	units['z'] = df.values[0, 0]
	units['value'] = df.values[0, 1]

	return value, z, phi, units


def write_csv(fpath: str, 
              value: np.ndarray, 
              z: np.ndarray, 
              phi: np.ndarray, 
              unitV: str, 
              unitZ: str):
    """
    Write image log data into a file in .csv format.

    Args:
        fpath (str): Output file path.
        value (np.ndarray): Log value.
        z (numpy.ndarray): Measured depth.
        phi (numpy.ndarray): Azimuth.
        unitV (str): Unit of log value.
        unitZ (str): Unit of measured depth.
    """
    # Column names.
    colName = ['Depth']
    
    # The first row contains units of measured depth and log value.
    unitRow = [unitZ]
    
    # Add new elements to column names and the unit row.
    for x in phi:
        colName.append(str(x))
        unitRow.append(unitV)
    
    # Create a dataframe of unit row.
    df0 = pd.DataFrame(data=[unitRow], columns=colName)
    
    # Create a dataframe of measured depth and log value.
    df1 = pd.DataFrame(data=np.c_[z, value], columns=colName)
    
    # Concatenate the two dataframes.
    df = pd.concat([df0, df1], ignore_index=True)
    
    # Write dataframe to file path.
    df.to_csv(fpath, index=False)
    
    
def crop_img_log(logval: np.ndarray, 
                 z: np.ndarray, 
                 nh: int = None, 
                 nw: int = None, 
                 verbose: bool = False):
    """
    Crop image log in measured depth direction.

    Args:
        logval (numpy.ndarray): Log value.
        z (numpy.ndarray): Measured depth.  
        nh (int | optional): Desired image log height (in sample numbers) after cropping.
                             If None, then it will be the height of the original image log. 
                             Defaults to None.
        nw (int | optional): Desired image log width (in sample numbers) after cropping.
                             If None, then it will be the width of the original image log. 
                             Defaults to None.
        verbose (bool): Whether to print the processing progress on screen.
                        Defaults to False. 

    Returns:
        output (list[dict]): The cropped image log.
                             {'value': log value, 'z': measured depth}. 
    """       
    # Initialize output.
    output = []
    
    # Get log length.
    log_h = logval.shape[0]
    log_w = logval.shape[1]
    
    # Output shape.
    if nh is None:
        nh = log_h
    if nw is None:
        nw = log_w
        
    # Crop image log.
    if verbose:
        sys.stdout.write('\rCropping image log...')
    i = 0
    stop = 0
    while(stop == 0):
        top = i * nh  # Top indice.
        bot = (i + 1) * nh  # Bottom indice.
        if bot >= log_h:
            bot = log_h
            stop = 1
        crop_logval = logval[top:bot, :]  # Cropped image log.
        crop_z = z[top:bot]  # Cropped measured depth.
        if log_w != nw:
            crop_logval = resize(crop_logval, (len(crop_logval), nw))
        output.append({'value': crop_logval, 'z': crop_z})
        i += 1
    
    if verbose:
        sys.stdout.write(' Done.\n')

    return output


def minmax_scale(x):
    """
    Scale data range to 0 and 1 by its minimum and maximum values.
    The data distribution will not be changed.

    Args:
        x (Numpy.ndarray): Input data.

    Returns:
        y (Numpy.ndarray): Scaled data.
    """
    min = np.nanmin(x)
    max = np.nanmax(x)
    y = (x - min) / (max - min)
    
    return y


def interp_nn(z, v, zi):
	"""
	Interpolate image log to new measured depths
	using nearest-neighbor interpolation.

	Args:
		z (numpy.ndarray): Original measured depth.
		v (numpy.ndarray): Image log
		zi (numpy.ndarray): New measured depth.
	"""
	# Initialize the interpolated log values.
	vi = np.zeros((len(zi), v.shape[-1]), dtype=v.dtype)

	# Infer sampling spacing from original measured depth.
	dzs = np.diff(z)

	# In case that the sampling spacing is not uniform, 
	# take its maxima.
	dz = np.amax(dzs)

	# Interpolation.
	for i in range(len(zi)):
		sys.stdout.write("\rInterpolating: %.2f%%" % 
						 ((i + 1) / len(zi) * 100))
		# Find the nearest neighbor of new measured depth 
		# in a depth interval equivalent to the sampling
		# spacing of original measured depth.
		
		# Depth interval.
		c = (z > zi[i] - dz / 2) & (z <= zi[i] + dz / 2)
		
		# Original measured depth in the interval.
		zit = z[c]

		# Original log values in the interval. 
		vit = v[c, :]

		# The nearest neighbor to the new measured depth.
		ind = np.argmin(np.abs(zit - zi[i]))

		# Take its log value.
		vi[i, :] = vit[ind, :]

	sys.stdout.write("\n")

	return vi


def split_consecutive(x: np.ndarray):
	"""
 	Split consecutive numbers.

	Args:
		x (np.ndarray): _description_

	Returns:
		_type_: _description_
	"""
	y = np.split(x, np.where(np.diff(x)!=1)[0]+1)

	return y


def split_circular_consecutive_indices(
    B: Sequence[int],
    n_cols: int,
) -> List[List[int]]:
    """
    Split 1D indices into consecutive groups on a circular axis [0, n_cols-1].

    Example:
        n_cols=144
        B=[0,1,2,5,6,7,141,142,143]
        -> [[141,142,143,0,1,2], [5,6,7]]

    Rules:
    - Indices are treated as points on a ring.
    - Consecutive means diff==1, and (n_cols-1)->0 is also consecutive.
    - Output groups are ordered such that:
        * normal groups: increasing
        * wrap group: tail increasing then head increasing (e.g., 141..143 + 0..2)
    """
    if n_cols <= 0:
        raise ValueError("n_cols must be positive")

    if len(B) == 0:
        return []

    # Normalize, unique, sort
    s = sorted({int(x) % n_cols for x in B})

    # Build linear groups by gaps>1
    groups = []
    cur = [s[0]]
    for x in s[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)

    # Merge wrap-around if needed: last group ends at n_cols-1 and first starts at 0
    if groups and groups[0][0] == 0 and groups[-1][-1] == n_cols - 1 and len(groups) > 1:
        wrap = groups[-1] + groups[0]  # tail then head
        groups = [wrap] + groups[1:-1]  # put wrap group first, keep others as-is

    # Optional: sort remaining groups by their first element for stable readability
    # (wrap group already first if exists)
    if groups:
        wrap_first = (groups[0][0] != 0 and groups[0][-1] != n_cols - 1)
        if wrap_first:  # no wrap group; sort all
            groups.sort(key=lambda g: g[0])
        else:
            # has wrap group at index 0; sort the rest
            rest = sorted(groups[1:], key=lambda g: g[0])
            groups = [groups[0]] + rest

    return groups


def resample(z, v, dz, method='average', verbose=True):
    """
    Resample image log.

    Args:
        z (Numpy.ndarray): Measured depth.
        v (Numpy.ndarray): Log value. Must be 2d array.
        dz (Float): Depth spacing after resampling.
        method (String): Resampling method. 
                         Options are:
                         1 - 'nearest': Take the nearest neighbor as the resampled value.
                         2 - 'average': Take the mean value in [z-dz/2, z+dz/2] as the resampled value.
                         3 - 'median': Take the median value in [z-dz/2, z+dz/2] as the resampled value.
                         4 - 'rms': Take the root-mean-squre value in [z-dz/2, z+dz/2] as the resampled value.
                         5 - 'circmean': Take the circular mean value in [z-dz/2, z+dz/2] as the resampled value.
                         Defaults to 'average'.

    Returns:
        znew (Numpy.ndarray): Resampled measured depth.
        vnew (Numpy.ndarray): Resampled log value.
    """
    # New Measured depth after resampling.
    znew = np.arange(start=np.amin(z) // dz * dz, 
                     stop=np.amax(z) // dz * dz + 2 * dz, 
                     step=dz, 
                     dtype=np.float64)
    
    # Initialize a new log value array, filled with NaNs.
    vnew = np.full((len(znew), v.shape[-1]), fill_value=np.nan, dtype=np.float64)
    
    # Resampling.
    for i in range(len(znew)):
        if verbose:
            sys.stdout.write('\rResampling: %.2f%%' % ((i+1) / len(znew) * 100))
        condition = (z > znew[i] - dz / 2) & (z <= znew[i] + dz / 2)  # Depth interval.
        index = np.argwhere(condition)  # Array-index in the depth interval.
        v_tmp = v[index]  # Logging value in the depth interval.
        z_tmp = z[index]  # Measured depth value in the depth interval.
        if len(v_tmp):  # If the depth interval contains log values.
            if method == 'nearest':  # Take the nearest neighbor.
                index_nn = np.argmin(np.abs(z_tmp - znew[i]))  # Array index of the nearest neighbor.
                vnew[i, :] = v_tmp[index_nn, :]
            if method == 'average':  # Take the average value.
                vnew[i, :] = np.nanmean(v_tmp, axis=0)
            if method == 'median':  # Take the median value.
                vnew[i, :] = np.nanmedian(v_tmp, axis=0)
            if method == 'rms':  # Take the root-mean-square value.
                vnew[i, :] = np.sqrt(np.nanmean(v_tmp**2, axis=0))
            if method == 'circmean':  # Take the circular mean value.
                v_tmp = np.deg2rad(v_tmp)
                vnew[i, :] = circmean(v_tmp, axis=0, high=math.pi, 
                                      nan_policy='omit')
                vnew[i, :] = np.rad2deg(vnew[i, :])
    if verbose:
        sys.stdout.write('\n')
    
    # Remove missing values.
    df = pd.DataFrame(data=np.c_[znew, vnew])
    # df.dropna(axis='index', how='any', inplace=True)
    # df.reset_index(drop=True, inplace=True)
    
    znew, vnew = df.values[:, 0], df.values[:, 1:] 
    
    return znew, vnew


def loop360(x):
	"""
	Loop the azimuth angle less than 0 or greater than 360.
	For example, azimuth angle -30 degree will be 330 degree, 
	and azimuth angle 390 degree will be 30 degree.

	Arg:
	x (numpy.ndarray or float): Azimuth angle [degree].

	Return:
	y (numpy.ndarray or float): Azimuth angle in [0, 360) degree. 
	"""
	if isinstance(x, np.ndarray):
		y = x.copy()
		y[y >= 360] = y[y >= 360] % 360
		y[y < 0] = y[y < 0] % 360
	else:
		y = x
		y = y % 360

	return y


def fill_nan(data):
    """
    Fill NaN values in a image log using the nearest neighbor interpolation.
    """
    # A boolean array indicating normal values (True) and NaN values (False).
    mask = ~np.isnan(data)
    # Coordinates of the array.
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    # Coordinates of normal values.
    xy = np.c_[xx[mask].ravel(), yy[mask].ravel()]
    # Interpolation.
    interp = NearestNDInterpolator(xy, data[mask].ravel())
    data_itp = interp(xx.ravel(), yy.ravel()).reshape(data.shape)
    
    return data_itp


def angular_diff(x1, x2):
    """
    Calculate angular/circular difference of two azimuths.
    """
    if x1 > 180:
        x1 -= 180
    if x2 > 180:
        x2 -= 180
    y = np.zeros(3, dtype=np.float32)
    y[0] = abs(x1 - x2)
    y[1] = abs(x1 + 180 - x2)
    y[2] = abs(x2 + 180 - x1)
    y = np.min(y)
    
    return y


def azimuth_diff(x1, x2):
    diff = abs(x1 - x2)
    if diff > 180:
        diff = 360 - diff
    
    return diff


def azimuth2angle(x):
	"""
	Convert the azimuthal angle clockwise from north direction (y-axis) ([0, 180] degree)
	to the counter-clockwise angle from x-axis ([0, 180] degree).

	Args:
		x (float or numpy.ndarray): Azimuthal angle clockwise from north direction [0, 180]. 

	Returns:
		y (float or numpy.ndarray): Counter-clockwise angle from x-axis [0, 180]. 
	"""
	if isinstance(x, np.ndarray):
		if (x > 180).any() or (x < 0).any():
			n = len(x[(x > 180) | (x < 0)])
			raise ValueError("The input angle array x must satisfy 0<=x<=180, "
							 "got %d elements out of range." % n)
		else:
			y = x.copy()
			y[(y >= 0) & (y <= 90)] = 90 - y[(y >= 0) & (y <= 90)]
			y[(y > 90) & (y <= 180)] = 270 - y[(y > 90) & (y <= 180)]
	
	else:
		if 0 <= x <= 90:
			y = 90 - x
		elif 90 < x <= 180:
			y = 270 - x
		else:
			raise ValueError("The input angle x must satisfy 0<=x<=180, got %.2f instead." % x)

	return y


def polar2cart(radius, angle):
	"""
	Convert polar coordinates to Cartesian coordinates.

	Args:
		radius (Numpy.ndarray): Radius
		angle (Numpy.ndarray): Azimuthal angle.

	Returns:
		x (Numpy.ndarray): X-coordinates.
		y (Numpy.ndarray): Y-coordinates.
	"""
	x = np.zeros(radius.shape, dtype=np.float64)
	y = np.zeros(radius.shape, dtype=np.float64)
	if radius.ndim == 2:
		for j in range(radius.shape[-1]):
			x[:, j] = radius[:, j] * math.sin(math.radians(angle[j]))
			y[:, j] = radius[:, j] * math.cos(math.radians(angle[j]))
	elif radius.ndim == 1:
		x = radius * np.sin(np.radians(angle))
		y = radius * np.cos(np.radians(angle))
	else:
		raise ValueError("Dimension of radius can only be 1 or 2, get %d instead" % radius.ndim)
	return x, y


def ellipse_axis_endpoints(xc, yc, d_major, d_minor, angle):
	"""
	Get ellipse axis endpoints coordinates.

	Args:
		xc (float): x-coordinate of the ellipse center.
		yc (float): y-coordinate of the ellipse center.
		d_major (float): major diameter.
		d_minor (float): minor diameter.
		angle (float): Counter-clockwise angle from x-axis to major axis [0, 180].

	Return:
		pts: (dictionary): Endpoint coordinates.
						   pts['major'] = [(x1_major, y1_major), 
						   				   (x2_major, y2_major)]
						   pts['minor'] = [(x1_minor, y1_minor), 
						   				   (x2_minor, y2_minor)]
	"""
	x1_major = xc + d_major/2 * math.cos(math.radians(angle))
	y1_major = yc + d_major/2 * math.sin(math.radians(angle))
	x2_major = xc - d_major/2 * math.cos(math.radians(angle))
	y2_major = yc - d_major/2 * math.sin(math.radians(angle))

	x1_minor = xc - d_minor/2 * math.sin(math.radians(angle))
	y1_minor = yc + d_minor/2 * math.cos(math.radians(angle))
	x2_minor = xc + d_minor/2 * math.sin(math.radians(angle))
	y2_minor = yc - d_minor/2 * math.cos(math.radians(angle))
	
	pts = {'major': [(x1_major, y1_major), (x2_major, y2_major)], 
		   'minor': [(x1_minor, y1_minor), (x2_minor, y2_minor)]}
	
	return pts


def elastic_transform_local(
    image: np.ndarray,        # [H, W]，float 或 uint8 等
    label: np.ndarray,        # [H, W]，二值(0/1)
    alpha: float = 8.0,       # 位移强度（像素）
    sigma: float = 6.0,       # 位移场平滑的高斯 σ（像素）
    mask_dilate: int = 5,     # 对正例区域膨胀的半径（像素）
    edge_sigma: float = 3.0,  # 软边的高斯 σ
    edge_gamma: float = 2.0,  # 软边幂次，>1 让位移更集中在内部
    circular_theta: bool = True,  # θ 维是否按圆周处理（水平 wrap）
    img_order: int = 1,       # 图像插值阶次：1=双线性，3=三次样条
    seed: int | None = None,
):
    """
    返回: (image_warp, label_warp)
    - image_warp: 与 image 同形状
    - label_warp: 二值化后的标签（0/1）
    """
    assert image.ndim == 2 and label.ndim == 2, "请输入二维数组 [H,W]"
    H, W = image.shape
    assert label.shape == (H, W), "label 尺寸必须与 image 一致"

    rng = np.random.default_rng(seed)

    # 1) soft mask：在正例及其邻域启用位移，边界平滑衰减
    mask = (label > 0.5).astype(np.float32)
    if mask_dilate > 0:
        k = 2 * mask_dilate + 1
        st = np.ones((k, k), dtype=bool)
        mask = binary_dilation(mask.astype(bool), structure=st).astype(np.float32)
    if edge_sigma > 0:
        soft = gaussian_filter(mask, sigma=edge_sigma, mode="nearest")
        if soft.max() > 0:
            soft = soft / soft.max()
    else:
        soft = mask
    if edge_gamma is not None and edge_gamma != 1.0:
        soft = soft ** edge_gamma
    # soft∈[0,1]，只在该区域产生位移
    soft = soft.astype(np.float32)

    # 2) 随机位移场（dz, dt），做高斯平滑并归一化到幅度 alpha
    dz = rng.normal(size=(H, W)).astype(np.float32)
    dt = rng.normal(size=(H, W)).astype(np.float32)
    if sigma > 0:
        dz = gaussian_filter(dz, sigma=sigma, mode="nearest")
        dt = gaussian_filter(dt, sigma=sigma, mode="nearest")

    norm = np.sqrt(dz * dz + dt * dt) + 1e-6
    dz = (dz / norm) * alpha * soft
    dt = (dt / norm) * alpha * soft

    # 3) 构造目标坐标（像素坐标系）
    zz, tt = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij"
    )
    z_new = zz + dz
    t_new = tt + dt

    # θ 维环形处理：t 坐标取模；z 维裁剪到合法范围
    if circular_theta:
        t_new = np.mod(t_new, W)
    else:
        t_new = np.clip(t_new, 0, W - 1)
    z_new = np.clip(z_new, 0, H - 1)

    # 4) 采样（图像双线性/三次，标签最近邻）
    # map_coordinates 接受 coords=[z, t]，形状 (2, H, W)
    coords = np.stack([z_new, t_new], axis=0)

    img_warp = map_coordinates(image.astype(np.float32), coords, order=img_order, mode="nearest")
    lbl_warp = map_coordinates(label.astype(np.float32), coords, order=0,        mode="nearest")
    lbl_warp = (lbl_warp > 0.5).astype(label.dtype)

    # 尝试保持原图数据类型
    if np.issubdtype(image.dtype, np.integer):
        img_warp = np.clip(np.rint(img_warp), np.iinfo(image.dtype).min, np.iinfo(image.dtype).max).astype(image.dtype)
    else:
        img_warp = img_warp.astype(image.dtype)

    return img_warp, lbl_warp


import numpy as np
from scipy.ndimage import map_coordinates

def _compose_homography(
    H: int, W: int,
    max_rot: float,                # 最大旋转（弧度）
    min_scale: float, max_scale: float,
    max_shear: float,              # 最大剪切（tan 值）
    max_persp: float,              # 透视分量上限（行3前两项）
    max_shift_z: float, max_shift_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """围绕图像中心生成随机的 3x3 单应矩阵（像素坐标系）"""
    C  = np.array([[1,0,-W/2],[0,1,-H/2],[0,0,1]], dtype=np.float32)
    Ci = np.array([[1,0, W/2],[0,1, H/2],[0,0,1]], dtype=np.float32)

    rot  = rng.uniform(-max_rot, max_rot)
    sx   = rng.uniform(min_scale, max_scale)
    sy   = rng.uniform(min_scale, max_scale)
    shx  = rng.uniform(-max_shear, max_shear)
    shy  = rng.uniform(-max_shear, max_shear)
    px   = rng.uniform(-max_persp, max_persp)
    py   = rng.uniform(-max_persp, max_persp)
    tx   = rng.uniform(-max_shift_t, max_shift_t)  # x 对应 theta(宽)
    ty   = rng.uniform(-max_shift_z, max_shift_z)  # y 对应 z(高)

    c, s = np.cos(rot), np.sin(rot)
    R  = np.array([[ c,-s, 0],[ s, c, 0],[ 0, 0, 1]], dtype=np.float32)
    S  = np.array([[sx, 0, 0],[ 0,sy, 0],[ 0, 0, 1]], dtype=np.float32)
    Sh = np.array([[ 1,shx,0],[shy, 1,0],[ 0, 0, 1]], dtype=np.float32)
    P  = np.array([[ 1, 0, 0],[ 0, 1, 0],[px,py, 1]], dtype=np.float32)
    T  = np.array([[ 1, 0, tx],[ 0, 1, ty],[ 0, 0,  1]], dtype=np.float32)

    # 围绕中心：Ci * T * R * Sh * S * P * C
    Hmat = Ci @ (T @ (R @ (Sh @ (S @ (P @ C)))))
    return Hmat


def perspective_transform(
    image: np.ndarray,        # [H,W] 或 [C,H,W] 或 [H,W,C]
    label: np.ndarray,        # [H,W]，二值 0/1
    *,
    # 若提供 homography，则使用之；否则按参数随机生成
    homography: np.ndarray | None = None,
    max_rot: float = 0.12,              # ≈ 7°
    min_scale: float = 0.9, max_scale: float = 1.1,
    max_shear: float = 0.08,
    max_persp: float = 0.0015,
    max_shift_z: float = 6.0,
    max_shift_t: float = 8.0,
    # 采样与环形条件
    circular_theta: bool = True,        # θ(宽) 方向环形
    img_order: int = 1,                 # 图像插值：1=双线性, 3=三次
    seed: int | None = None,
):
    """
    返回: (image_warp, label_warp)

    - 对整张图像/标签施加同一透视变换；
    - θ 方向可环形 wrap，仅对宽度生效；z 方向裁剪到边界；
    - 图像用双线性插值，标签最近邻后阈值回二值。
    """
    # --- 规范形状 ---
    if image.ndim == 2:
        H, W = image.shape
        img_is_chlast = True
        get_ch  = lambda x: 1
        get_hw  = lambda x: x.shape
        set_img = lambda arr: arr
        img_channels = 1
    elif image.ndim == 3:
        # 支持 [C,H,W] 或 [H,W,C]
        if image.shape[0] in (1,3,4) and image.shape[0] < min(image.shape[1], image.shape[2]):
            # 认为是 [C,H,W]
            C, H, W = image.shape
            img_is_chlast = False
            img_channels = C
            get_ch  = lambda x: x.shape[0]
            get_hw  = lambda x: x.shape[1], x.shape[2]
            set_img = lambda arr: arr
        else:
            # 认为是 [H,W,C]
            H, W, C = image.shape
            img_is_chlast = True
            img_channels = C
            get_ch  = lambda x: x.shape[-1]
            get_hw  = lambda x: x.shape[0], x.shape[1]
            set_img = lambda arr: arr
    else:
        raise ValueError("image must be [H,W] or [C,H,W] or [H,W,C]")

    assert label.shape == (H, W), "label must be [H,W] and match image H,W"

    rng = np.random.default_rng(seed)

    # --- 单应矩阵 ---
    if homography is None:
        Hmat = _compose_homography(
            H, W, max_rot, min_scale, max_scale, max_shear, max_persp, max_shift_z, max_shift_t, rng
        )
    else:
        Hmat = np.asarray(homography, dtype=np.float32)
        assert Hmat.shape == (3,3)
    Hinv = np.linalg.inv(Hmat)

    # --- 目标网格与逆映射 ---
    zz, tt = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij"
    )
    X = tt.reshape(-1)
    Y = zz.reshape(-1)
    den = Hinv[2,0]*X + Hinv[2,1]*Y + Hinv[2,2]
    xs  = (Hinv[0,0]*X + Hinv[0,1]*Y + Hinv[0,2]) / den  # 对应 θ 轴(宽度)
    ys  = (Hinv[1,0]*X + Hinv[1,1]*Y + Hinv[1,2]) / den  # 对应 z 轴(高度)
    xs = xs.reshape(H, W)
    ys = ys.reshape(H, W)

    # --- 环形和裁剪 ---
    if circular_theta:
        xs = np.mod(xs, W)
    else:
        xs = np.clip(xs, 0, W-1)
    ys = np.clip(ys, 0, H-1)

    coords = np.stack([ys, xs], axis=0)  # map_coordinates 需要 [y, x]

    # --- 采样图像 ---
    if image.ndim == 2:
        img_warp = map_coordinates(image.astype(np.float32), coords, order=img_order, mode="nearest")
        # 回到原 dtype
        img_warp = img_warp.astype(image.dtype) if image.dtype.kind != 'f' else img_warp
    else:
        # 多通道：逐通道同一坐标采样
        if img_is_chlast:  # [H,W,C]
            img_warp = np.empty_like(image, dtype=np.float32)
            for c in range(img_channels):
                img_warp[..., c] = map_coordinates(image[..., c].astype(np.float32), coords, order=img_order, mode="nearest")
        else:              # [C,H,W]
            img_warp = np.empty_like(image, dtype=np.float32)
            for c in range(img_channels):
                img_warp[c, ...] = map_coordinates(image[c].astype(np.float32), coords, order=img_order, mode="nearest")
        # 回 dtype
        if image.dtype.kind != 'f':
            img_warp = np.clip(np.rint(img_warp),
                               np.iinfo(image.dtype).min,
                               np.iinfo(image.dtype).max).astype(image.dtype)

    # --- 采样标签（最近邻）并阈值 ---
    lbl_warp = map_coordinates(label.astype(np.float32), coords, order=0, mode="nearest")
    lbl_warp = (lbl_warp > 0.5).astype(label.dtype)

    return img_warp, lbl_warp


def blocky_artifact(
    img: np.ndarray,
    label: np.ndarray,
    n_groups: int = 3,
    m0: int = 3,
    m1: int = 6,
    max_shift_px: int = 8,
    mosaic_w: int = 8,
    mosaic_h: Optional[int] = None,   # None 则使用该组高度 m_i
    non_overlapping: bool = True,
    force_nonzero_shift: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    在 z 轴上随机选 n 组行；每组行数 m_i ∈ [m0, m1]。
    每组在 θ 方向做【同一方向、同一幅度】的圆周平移：
      - 图像：roll + 马赛克降质（块均值）
      - 标签：仅 roll（不降质、不置零）

    参数:
      img, label: 形状 (H, W) 的二维数组，且形状一致
      n_groups: 组数
      m0, m1: 每组的行数范围（闭区间）
      max_shift_px: 每组最大平移像素（θ 方向）
      mosaic_w: 水平马赛克块宽
      mosaic_h: 垂直马赛克块高；None 则用该组高度 m_i
      non_overlapping: True 则各组行段不重叠
      force_nonzero_shift: True 则强制每组 shift != 0
      seed: 随机种子（可复现）

    返回:
      img_out, label_out, ops
    """
    if img.ndim != 2 or label.ndim != 2:
        raise ValueError("img 和 label 必须是二维数组 (H, W)")
    if img.shape != label.shape:
        raise ValueError("img 和 label 形状必须一致")

    H, W = img.shape
    if H < 1 or W < 1:
        return img.copy(), label.copy(), []
    if not (isinstance(m0, int) and isinstance(m1, int) and 1 <= m0 <= m1):
        raise ValueError("m0, m1 必须是正整数且满足 1 <= m0 <= m1")
    if H < m0:
        return img.copy(), label.copy(), []

    rng = np.random.default_rng(seed)
    img_out = img.copy()
    label_out = label.copy()

    # 候选起点（至少能放下 m0 行）
    starts_all = np.arange(0, H - m0 + 1, dtype=int)
    rng.shuffle(starts_all)

    chosen: List[Tuple[int, int]] = []  # (start, m_i)
    occupied = np.zeros(H, dtype=bool)

    # 选组（可选不重叠）
    for s in starts_all:
        if len(chosen) >= n_groups:
            break
        max_len_here = min(m1, H - s)
        if max_len_here < m0:
            continue
        for _ in range(5):
            m_i = int(rng.integers(m0, max_len_here + 1))
            if non_overlapping and occupied[s:s + m_i].any():
                continue
            chosen.append((s, m_i))
            if non_overlapping:
                occupied[s:s + m_i] = True
            break

    if len(chosen) == 0:
        return img_out, label_out, []

    mosaic_w = max(1, int(mosaic_w))
    ops: List[Dict] = []

    for (r0, m_i) in chosen:
        r1 = min(r0 + m_i, H)
        rows = np.arange(r0, r1, dtype=int)

        # 该组统一的 shift
        if max_shift_px > 0:
            shift = int(rng.integers(-max_shift_px, max_shift_px + 1))
            if force_nonzero_shift:
                while shift == 0:
                    shift = int(rng.integers(-max_shift_px, max_shift_px + 1))
        else:
            shift = 0

        # 1) 行级圆周平移（θ 向，统一 shift）
        if shift != 0:
            img_out[r0:r1, :] = np.roll(img_out[r0:r1, :], shift, axis=1)
            label_out[r0:r1, :] = np.roll(label_out[r0:r1, :], shift, axis=1)

        # 2) 马赛克降质（仅图像；按块均值）
        blk_h = m_i if (mosaic_h is None or mosaic_h <= 0) else max(1, int(mosaic_h))
        rr = r0
        while rr < r1:
            r_end = min(rr + blk_h, r1)
            for cc in range(0, W, mosaic_w):
                c_end = min(cc + mosaic_w, W)
                block = img_out[rr:r_end, cc:c_end]
                val = np.nanmean(block)
                if not np.isnan(val):
                    img_out[rr:r_end, cc:c_end] = val
            rr += blk_h

    return img_out, label_out


def _block_reduce_mean(a: np.ndarray, bz: int, bt: int, circular_theta: bool, pad_mode_z: str) -> np.ndarray:
    """非重叠块平均降采样（支持 θ 维 wrap），返回下采样后的数组。"""
    H, W = a.shape
    Hb = int(np.ceil(H / bz)) * bz
    Wb = int(np.ceil(W / bt)) * bt
    pad_z = Hb - H
    pad_t = Wb - W
    if pad_t > 0:
        # θ 维：wrap 表示环向循环
        a = np.pad(a, ((0, 0), (0, pad_t)), mode="wrap" if circular_theta else "edge")
    if pad_z > 0:
        a = np.pad(a, ((0, pad_z), (0, 0)), mode=pad_mode_z)
    a = a.reshape(Hb // bz, bz, Wb // bt, bt)
    return a.mean(axis=(1, 3))


def _block_reduce_median(a: np.ndarray, bz: int, bt: int, circular_theta: bool, pad_mode_z: str) -> np.ndarray:
    H, W = a.shape
    Hb = int(np.ceil(H / bz)) * bz
    Wb = int(np.ceil(W / bt)) * bt
    pad_z = Hb - H
    pad_t = Wb - W
    if pad_t > 0:
        a = np.pad(a, ((0, 0), (0, pad_t)), mode="wrap" if circular_theta else "edge")
    if pad_z > 0:
        a = np.pad(a, ((0, pad_z), (0, 0)), mode=pad_mode_z)
    a = a.reshape(Hb // bz, bz, Wb // bt, bt)
    return np.median(a, axis=(1, 3))


def _block_reduce_mode(lbl: np.ndarray, bz: int, bt: int, circular_theta: bool, pad_mode_z: str) -> np.ndarray:
    """非重叠块众数（适合整型标签）。对浮点 0/1 标签会自动取四舍五入。"""
    if lbl.dtype.kind == 'f':
        lbl = np.rint(lbl).astype(np.int64)
    else:
        lbl = lbl.astype(np.int64)
    H, W = lbl.shape
    Hb = int(np.ceil(H / bz)) * bz
    Wb = int(np.ceil(W / bt)) * bt
    pad_z = Hb - H
    pad_t = Wb - W
    if pad_t > 0:
        lbl = np.pad(lbl, ((0, 0), (0, pad_t)), mode="wrap" if circular_theta else "edge")
    if pad_z > 0:
        lbl = np.pad(lbl, ((0, pad_z), (0, 0)), mode=pad_mode_z)
    hb, wb = Hb // bz, Wb // bt
    out = np.empty((hb, wb), dtype=np.int64)
    # 简单双循环（块数量通常不大，易读且稳定）
    for i in range(hb):
        for j in range(wb):
            block = lbl[i*bz:(i+1)*bz, j*bt:(j+1)*bt].ravel()
            vals, counts = np.unique(block, return_counts=True)
            out[i, j] = vals[counts.argmax()]
    return out


def mosaic_degrade(
    img: np.ndarray,
    lbl: np.ndarray,
    z_range: Tuple[int, int] = (2, 8),
    t_range: Tuple[int, int] = (2, 16),
    img_reduce: str = "mean",          # "mean" | "median"
    lbl_reduce: str = "mode",           # "mode" | "nearest"（nearest 等价于先下采样索引再上采样）
    circular_theta: bool = True,        # θ 维是否环向
    pad_mode_z: str = "edge",          # z 维 pad 模式："edge"/"reflect"/"constant"
    prob: float = 1.0,                  # 应用概率
    rng: Optional[np.random.Generator] = None,
):
    """
    对整张成像测井图像及其标签进行“马赛克化（像素化）”降质，
    模拟 z（轴向）与 θ（角向）的采样精度不足。

    输入
    ----
    img: [Z, T] 的浮点图像（可多通道则先外部循环）
    lbl: [Z, T] 的标签（整型或 0/1 浮点）
    z_range, t_range: 随机块尺寸范围（含端点），单位像素
    img_reduce: 图像块内聚合方式（mean/median）
    lbl_reduce: 标签聚合方式（mode 或 nearest）。mode 更鲁棒，nearest 更锐利。
    circular_theta: θ 维是否环向（True 则使用 wrap 填充）
    pad_mode_z: z 维填充方式（采样对齐时使用）
    prob: 本增强被应用的概率（<1 时可能直接返回原图）

    返回
    ----
    img_out, lbl_out: 与输入同形状的像素化图像与标签
    """
    assert img.shape == lbl.shape and img.ndim == 2, "img/lbl 必须同形状二维数组 [Z,T]"
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > prob:
        return img, lbl

    z0, z1 = z_range
    t0, t1 = t_range
    assert z0 >= 1 and t0 >= 1 and z1 >= z0 and t1 >= t0
    bz = int(rng.integers(z0, z1 + 1))
    bt = int(rng.integers(t0, t1 + 1))

    # 1) 下采样（块聚合）
    if img_reduce == "median":
        img_low = _block_reduce_median(img, bz, bt, circular_theta, pad_mode_z)
    else:
        img_low = _block_reduce_mean(img, bz, bt, circular_theta, pad_mode_z)

    if lbl_reduce == "nearest":
        # 通过取块左上角索引近似 nearest（比 mode 更保持边界但对噪声敏感）
        # 实现方式：在下采样阶段取样本点，而非统计
        H, W = img.shape
        hb, wb = int(np.ceil(H / bz)), int(np.ceil(W / bt))
        # 计算索引网格（考虑 θ wrap 和 z pad）
        zi = np.minimum(np.arange(hb) * bz, H - 1)
        tj = (np.arange(wb) * bt) % W if circular_theta else np.minimum(np.arange(wb) * bt, W - 1)
        lbl_low = lbl[zi[:, None], tj[None, :]]
    else:
        lbl_low = _block_reduce_mode(lbl, bz, bt, circular_theta, pad_mode_z)

    # 2) 上采样（repeat 回原尺寸）
    img_px = np.repeat(np.repeat(img_low, bz, axis=0), bt, axis=1)
    lbl_px = np.repeat(np.repeat(lbl_low, bz, axis=0), bt, axis=1)

    # 3) 裁剪回原尺寸
    H, W = img.shape
    img_px = img_px[:H, :W]
    lbl_px = lbl_px[:H, :W]

    return img_px.astype(img.dtype, copy=False), lbl_px.astype(lbl.dtype, copy=False)


def zcrop_stretch(
    img: np.ndarray,
    lbl: np.ndarray,
    require_positive: bool = True,
    img_order: int = 1,          # 图像插值阶数：1=bilinear
    lbl_order: int = 0,          # 标签插值阶数：0=nearest
    sub_order: int = 2,          # 裁剪后的图像长度是原图像的1/sub_order
    anti_aliasing_img: bool = True,
    keep_label_binary: bool = True, 
    pos_thr = 0.5, 
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机沿 Z 轴裁掉一定高度，确保裁剪窗内有崩落（label>pos_thr），
    然后把裁剪结果沿 Z 拉伸回原始高度。T（θ向）尺寸保持不变。

    img, lbl: 形状 [Z, T] 的二维数组；lbl 可为 0/1 浮点或整型掩码。
    prob: 应用该增强的概率；若随机未命中则原样返回。
    require_positive: 若为 True 且整幅 lbl 无正样本，则原样返回。
    pos_thr: 判定正样本的阈值（对浮点标签有效）。
    """
    assert img.shape == lbl.shape and img.ndim == 2, "img/lbl 必须同形状二维数组 [Z, T]"
    Z, T = img.shape
    if Z < 2:
        return img, lbl

    rng = np.random.default_rng(seed)

    # 要裁剪的高度 = Z//sub_order
    z_len = max(1, Z // sub_order)
    start_max = Z - z_len

    # 计算哪些起点能使裁剪窗内至少包含一个正样本
    pos_any_row = (lbl > pos_thr).any(axis=1).astype(np.int32)  # [Z]
    if pos_any_row.sum() == 0:
        if require_positive:
            # 无正样本且必须包含 => 放弃增强
            return img, lbl
        # 否则随便裁一半
        start = int(rng.integers(0, start_max + 1))
    else:
        # 用前缀和快速判断每个窗内是否有正样本
        prefix = np.concatenate([[0], np.cumsum(pos_any_row)])  # 长度 Z+1
        # 窗口 [s, s+z_len) 内的正样本数
        window_cnt = prefix[z_len:] - prefix[:-z_len]          # 长度 Z - z_len + 1
        valid_starts = np.where(window_cnt > 0)[0]
        if valid_starts.size == 0:
            # 理论上不会发生，但做个保护
            start = int(rng.integers(0, start_max + 1))
        else:
            start = int(rng.choice(valid_starts))
    end = start + z_len

    # 裁剪
    img_crop = img[start:end, :]
    lbl_crop = lbl[start:end, :]

    # 拉伸回原高（仅缩放 Z 维；T 不变）
    img_out = resize(
        img_crop, (Z, T),
        order=img_order, mode="edge",
        anti_aliasing=anti_aliasing_img,
        preserve_range=True,
    ).astype(img.dtype, copy=False)

    lbl_out = resize(
        lbl_crop, (Z, T),
        order=lbl_order, mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    if keep_label_binary:
        lbl_out = (lbl_out >= pos_thr).astype(lbl.dtype, copy=False)
    else:
        lbl_out = lbl_out.astype(lbl.dtype, copy=False)

    return img_out, lbl_out


def plot_atv_log(
    log, 
    log_type="Amplitude", 
    cmap="afmhot", 
    save_path="./amp.png", 
    vmin=None, 
    vmax=None, 
    use_clim=False, 
    use_ax=None
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if use_clim:
        vmin, vmax = log['clim']
    if use_ax is None:
        plt.figure()
        im = plt.imshow(log["data"], cmap=cmap, extent=[0, 360, log["depth"].max(), log["depth"].min()], aspect='auto', vmin=vmin, vmax=vmax)
        plt.xticks([0, 90, 180, 270, 360])
        plt.xlabel('Azimuth (°)')
        plt.ylabel('Depth (m)')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.08) 
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(f'{log_type} (%s)' % log["data_unit"])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        ax = use_ax
        im = ax.imshow(log["data"], cmap=cmap, extent=[0, 360, log["depth"].max(), log["depth"].min()], aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel('Azimuth (°)')
        ax.set_ylabel('Depth (m)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.08) 
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(f'{log_type} (%s)' % log["data_unit"])


def plot_fmi_log(log, cmap="afmhot_r", save_path="./log.png"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.figure()
    im = plt.imshow(log["data"], cmap=cmap, extent=[0, 360, log["depth"].max(), log["depth"].min()], aspect='auto')
    plt.xticks([0, 90, 180, 270, 360])
    plt.xlabel('Azimuth (°)')
    plt.ylabel('Depth (m)')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.08) 
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Log value (%s)' % log["data_unit"])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_atv_log_mask(
    log, 
    mask, 
    log_type="Amplitude", 
    cmap="gray", 
    mask_color="red", 
    save_path="./amp_label.png", 
    vmin=None, 
    vmax=None, 
    use_clim=False, 
    use_ax=None
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if mask_color in ["red", "r"]:
        color = np.array([255/255, 0/255, 0/255, 0.3])
    elif mask_color in ["blue", "b"]:
        color = np.array([0/255, 0/255, 255/255, 0.3])
    elif mask_color in ["green", "g"]:
        color = np.array([0/255, 255/255, 0/255, 0.3])
    else:
        raise ValueError("Does not support the color %s, choose 'red', 'green', or 'blue'" % mask_color)
    mask_img = mask["data"].reshape(mask["data"].shape[0], mask["data"].shape[1], 1) * color.reshape(1, 1, -1)
    if use_clim:
        vmin, vmax = log["clim"]
    if use_ax is None:
        plt.figure()
        im = plt.imshow(log["data"], cmap=cmap, extent=[0, 360, log["depth"].max(), log["depth"].min()], aspect='auto', vmin=vmin, vmax=vmax)
        plt.imshow(mask_img, extent=[0, 360, mask["depth"].max(), mask["depth"].min()], aspect='auto')
        plt.xticks([0, 90, 180, 270, 360])
        plt.xlabel('Azimuth [°]')
        plt.ylabel('Depth [m]')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.08) 
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(f'{log_type} (%s)' % log["data_unit"])
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        ax = use_ax
        im = ax.imshow(log["data"], cmap=cmap, extent=[0, 360, log["depth"].max(), log["depth"].min()], aspect='auto', vmin=vmin, vmax=vmax)
        ax.imshow(mask_img, extent=[0, 360, mask["depth"].max(), mask["depth"].min()], aspect='auto')
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel('Azimuth [°]')
        ax.set_ylabel('Depth [m]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.08) 
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(f'{log_type} (%s)' % log["data_unit"])


def plot_fmi_log_mask(log, mask, cmap="gray_r", 
                      mask_color=["black", "red"], 
                      mask_alpha=0.4, 
                      save_path="./log_label.png"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    lut = np.stack([
            rgba(c, alpha=0.0 if i == 0 else mask_alpha)
            for i, c in enumerate(mask_color)
        ], axis=0)
    mask_img = lut[mask["data"]]

    plt.figure()
    im = plt.imshow(log["data"], cmap=cmap, extent=[0, 360, log["depth"].max(), log["depth"].min()], aspect='auto')
    plt.imshow(mask_img, extent=[0, 360, mask["depth"].max(), mask["depth"].min()], aspect='auto')
    plt.xticks([0, 90, 180, 270, 360])
    plt.xlabel('Azimuth [°]')
    plt.ylabel('Depth [m]')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('right', size='5%', pad=0.08) 
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Log value (%s)' % log["data_unit"])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_uint8_png_no_axes(img_uint8, path_png, path_npy=None):
    Image.fromarray(img_uint8.astype(np.uint8), mode="L").save(path_png)
    if path_npy is not None:
        np.save(path_npy, img_uint8.astype(np.uint8))


def rgba(name, alpha=0.3):
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(name)
    return np.array([r, g, b, alpha], float)


if __name__ == "__main__":
    B = [0,1,2,3,5,6,7,141,142,143]
    s = split_circular_consecutive_indices(B, n_cols=144)
    print(s)
    for j in range(len(s)):
        print(np.median(s[j]))

