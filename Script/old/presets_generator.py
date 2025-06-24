
### Code to randomly generate N_TESTS presets named test_x.csv,
### with parameters changing [-RANGE, +RANGE] respect to the given values inserted into params.

import pandas as pd
import os
from sklearn.model_selection import ParameterSampler


N_TESTS = 10  # adjust as needed
RANGE = 3  # adjust as needed



# Original parameters
params = {
    'seg_level': -1,
    'sthresh': 15,
    'mthresh': 11,
    'close': 2,
    'use_otsu': ['FALSE', 'TRUE'],
    'a_t': 1,
    'a_h': 1,
    'max_n_holes': 2,
    'vis_level': -1,
    'line_thickness': 50,
    'white_thresh': 5,
    'black_thresh': 50,
    'use_padding': ['FALSE', 'TRUE'],
    'contour_fn': ['four_pt', 'five_pt', 'simple'],
    'keep_ids': ['none'],
    'exclude_ids': ['none'],
}

def generate_range(val):
    if isinstance(val, int):
        # Preserve order: val-3 to val+3 inclusive
        return [val + i for i in range(-RANGE, 1+RANGE)]
    return val  # categorical parameters remain unchanged

# Build parameter distribution dictionary with ranges/lists
param_dist = {k: generate_range(v) for k, v in params.items()}

# Output directory (make sure to use exact casing)
output_dir = "MLIAProject/CLAM/presets"
os.makedirs(output_dir, exist_ok=True)

# Number of random samples to generate

# Keep keys order (important!)
keys = list(param_dist.keys())

# Sample random combinations
sampler = list(ParameterSampler(param_dist, n_iter=N_TESTS, random_state=42))

# Save each combination as CSV with columns in correct order
for idx, combo in enumerate(sampler):
    df = pd.DataFrame([combo], columns=keys)
    df.to_csv(os.path.join(output_dir, f'test_{idx}.csv'), index=False)

print(f"Generated {len(sampler)} test CSV files in '{output_dir}'.")
