# Prepare training data for ocean ML

## 1. Super-Resolution Sea Surface Salinity (SSS) Data Retrieval

This project provides tools and data descriptions for working with 2D surface field from a 1/48-degree MITgcm simulation (llc4320). The goal is to use these super-high resolution 2D field to create training dataset for machine learning training. In the example, we target on super resolution. This problem is motivated by the need of higher resolution salinity map beyond the current remote sensing capability. 

### Data Description

#### Global SST/SSS Data from llc4320
- **Added**: June 29, 2024
- **Location**: The original llc4320 data is stored in NASA AMES HPC.  
- **Details**: 
  - Each file contains SST data from a 1/48-degree MITgcm simulation (~2 km resolution).
  - If 'shrunk' is in the filename, the data is saved unstructured after removing zero values over continents.
  - A mask file (`hFacC_k0.data`) identifies "wet" cells (where `hfacc != 0`), which can be used to restore the full model grid from the shrunk data.
- **File Format**: Binary (big-endian float32, `>f4`)
- **Resolution**: Original grid is 12960 x 8640, split into eastern and western hemispheres (8640 x 12960 each).

#### Utilities
- **File**: `utils.py`
- **Purpose**: Contains the `mds2d` function to map shrunk 2D data into two 2D fields (eastern and western hemispheres).
- **Usage**: See the example below.

### Requirements

To use this data and code, install the following Python packages:
```bash
pip install earthaccess matplotlib xarray requests numpy scipy
```

### Usage Example

#### Loading and Processing SSS Data
```python
import numpy as np
import utils

# Load mask and SSS data
pth='/home/jovyan/shared-public/swot_shared/super_resolution/SSS'
mask = np.fromfile(f'{pth}/hFacC_k0.data', '>f4')
sss_shrunk = np.fromfile(f'{pth}/SSS.0001400112.shrunk', '>f4')

# Restore full grid: apply shrunk data to wet cells, set dry cells to NaN
mask[mask != 0] = sss_shrunk
mask[mask == 0] = np.nan

# Split into 2D fields (east and west hemispheres)
sss_east, sss_west = utils.mds2d(mask)
print(sss_east.shape, sss_west.shape)  # Output: (12960, 8640) (8640, 12960)
```

#### Visualizing the Data
```python
import matplotlib.pyplot as plt

# Plot eastern and western hemispheres
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(sss_east, cmap='viridis')
ax[0].set_title('Eastern Hemisphere SSS')
ax[1].imshow(sss_west, cmap='viridis')
ax[1].set_title('Western Hemisphere SSS')
plt.show()
```

## Notes
- **File Paths**: Ensure the data files are accessible at the specified paths. If using a different system, adjust the paths accordingly.
- **Storage**: The "shrunk" format reduces file size, but reconstructing the full grid requires the mask file.
- **Performance**: Processing large grids (12960 x 8640) may require significant memory; consider subsetting data for smaller analyses.

## History
- **06/29/2024**: 
  - Uploaded initial SSS data description to `/home/jovyan/shared-public/swot_shared/super_resolution/SSS/`.
  - Added `utils.py` for data loading and processing.

## Contributing
Feel free to contribute by submitting pull requests or reporting issues. For questions, contact the project maintainers via the shared repository.

## License
This code is on MITgcm license. Please also ensure compliance with any applicable data-sharing agreements.

## Funding
This project is funded by NASA Physical Oceanography Program as a part of the Ocean AI working group activity.
