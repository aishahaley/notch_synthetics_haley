#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from pathlib import Path
import numpy as np
import dask.array as da
from dask_image.ndfilters import gaussian_filter
import sys
from iohub import open_ome_zarr

if __name__ == "__main__":
    path_to_zarr = r'/mnt/Data3/Haley/Garcia_2025/2025-01-13/12CSL-jffx673-his-RFP/12CSL-jffx673-his-RFP_combined.zarr'
    path_to_new_zarr = Path(r'/mnt/Data3/Haley/Garcia_2025/2025-01-13/12CSL-jffx673-his-RFP/12CSL-jffx673-his-RFP_combined_blur.ome.zarr')
    with open_ome_zarr(path_to_zarr, mode= "r", layout="auto",) as dataset:

        channel_names = dataset.channel_names
        print(channel_names)
        images = dataset["0"]
        print(images.shape)
        sigma = (0.3, 5, 5)
        timepoints = images.shape[0]

        out_ds = open_ome_zarr(path_to_new_zarr, layout="fov", mode="a", channel_names=["H2b Blur"])
        for timepoint in tqdm(range(timepoints)):
            im = da.array(images[timepoint])[0]
            im_blur = gaussian_filter(im, sigma)
            # Reshape from (z, y, x) to (t, c, z, y, x)
            im_blur = da.expand_dims(im_blur, axis=0)
            im_blur = da.expand_dims(im_blur, axis=0)
            im_blur = np.asarray(im_blur)
            if timepoint == 0:
                out_ds["img"] = im_blur
            else:
                out_ds["img"].append(im_blur, axis=0)

