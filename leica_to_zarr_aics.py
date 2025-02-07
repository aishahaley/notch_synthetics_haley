# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 2025 By Haley Bianchi

@author: aishahaley
adapted from code written by Vinu Harihar, Bruno Moretti and Yovan Badal
"""
import sys
import os
import numpy as np
#import pims
import zarr
import ome_zarr
from ome_zarr.writer import write_image
from pathlib import Path
from typing import List
from natsort import natsorted
from tqdm import tqdm
from iohub.ngff import open_ome_zarr
import aicsimageio
from aicsimageio.dimensions import Dimensions
import dask.array as da
import glob

def list_lif(folder: Path) -> List:
    """
    function uses the glob library to return an iterable of Paths from the czi files in the data folder
    """
    return list(folder.glob('*.lif'))


def main():
    path_to_lif = Path(sys.argv[1])
    out_path = path_to_lif / "zarr"
    out_path.mkdir(exist_ok=True)
    file_list = list_lif(path_to_lif)
    file_list = natsorted(file_list)

    for file in tqdm(file_list):
        lif_reader = aicsimageio.readers.LifReader(file)
        print("Lif file metadata {}".format(lif_reader.metadata))

        for scene_idx, scene_name in enumerate(lif_reader.scenes):
            scene_data = lif_reader.get_image_dask_data(S=scene_idx).compute()
            print(f"Original scene shape: {scene_data.shape}")

            # Ensure 5D data (t, c, z, y, x)
            if scene_data.ndim == 4:  # If missing time dimension
                scene_data = np.expand_dims(scene_data, axis=1)
                print(f"Adjusted scene shape: {scene_data.shape}")

            scene_path = Path(out_path) / f"scene_{scene_idx}.zarr"
            store = zarr.DirectoryStore(scene_path)
            root = zarr.group(store=store, overwrite=True)

            # Store the image data first
            z_array = root.create_dataset(
                '0',
                data=scene_data,
                chunks=scene_data.shape,
                dtype=scene_data.dtype
            )
            pixel_size = lif_reader.physical_pixel_sizes

            # Create scale array to match data dimensions exactly
            scale = [1.0] * scene_data.ndim  # Initialize with number of dimensions in data
            # Set spatial scales for the last three dimensions
            scale[-3:] = [float(pixel_size.Z), float(pixel_size.Y), float(pixel_size.X)]
            print(f"Scale array: {scale}")

            root.attrs['multiscales'] = [{
                'version': '0.4',
                'name': scene_name,
                'datasets': [{
                    'path': '0',
                    'coordinate_transformations': [{
                        'type': 'scale',
                        'scale': scale
                    }]
                }],
                'axes': [
                            {'name': 't', 'type': 'time'},
                            {'name': 'c', 'type': 'channel'},
                            {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                            {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
                        ][:scene_data.ndim]  # Limit axes to match data dimensions
            }]

            root.attrs['omero'] = {
                'id': 1,
                'name': scene_name,
                'version': '0.4',
                'channels': [{'label': ch} for ch in lif_reader.channel_names]
            }

            # Verify using shape instead of scale
            opened_zarr = open_ome_zarr(scene_path)
            print(f"Converted {scene_name}: {opened_zarr.scale}")


def combine():
    path_to_lif = Path(sys.argv[1])
    out_path = path_to_lif / "zarr"
    out_path.mkdir(exist_ok=True)
    file_list = list_lif(path_to_lif)
    file_list = natsorted(file_list)


    for file in tqdm(file_list):
        lif_reader = aicsimageio.readers.LifReader(file)
        print("Lif file metadata {}".format(lif_reader.metadata))

        # First pass to determine maximum dimensions
        max_c = max_z = max_y = max_x = 0
        total_timepoints = 0  # Will combine scenes and timepoints
        scene_shapes = []

        for scene_idx in range(len(lif_reader.scenes)):
            scene_data = lif_reader.get_image_dask_data(S=scene_idx).compute()
            if scene_data.ndim == 4:
                scene_data = np.expand_dims(scene_data, axis=0)
            scene_shapes.append(scene_data.shape)
            total_timepoints += scene_data.shape[0]  # Add timepoints from this scene
            max_c = max(max_c, scene_data.shape[1])
            max_z = max(max_z, scene_data.shape[2])
            max_y = max(max_y, scene_data.shape[3])
            max_x = max(max_x, scene_data.shape[4])

        # Create a single zarr file with 5D structure (T, C, Z, Y, X)
        combined_shape = (total_timepoints, max_c, max_z, max_y, max_x)
        combined_path = Path(out_path) / f"{file.stem}_combined.zarr"
        store = zarr.DirectoryStore(combined_path)
        root = zarr.group(store=store, overwrite=True)

        # Create dataset with combined shape
        combined_data = root.create_dataset(
            '0',
            shape=combined_shape,
            chunks=(1, max_c, max_z, max_y, max_x),  # Chunk by timepoint
            dtype=scene_data.dtype,
            fill_value=0
        )

        # Copy each scene into the combined array
        current_t = 0
        for scene_idx in range(len(lif_reader.scenes)):
            scene_data = lif_reader.get_image_dask_data(S=scene_idx).compute()
            if scene_data.ndim == 4:
                scene_data = np.expand_dims(scene_data, axis=0)

            # Get dimensions of this scene
            t, c, z, y, x = scene_data.shape

            # Copy data
            combined_data[current_t:current_t + t, :c, :z, :y, :x] = scene_data
            current_t += t

        pixel_size = lif_reader.physical_pixel_sizes
        scale = [1.0, 1.0, float(pixel_size.Z), float(pixel_size.Y), float(pixel_size.X)]

        root.attrs['multiscales'] = [{
            'version': '0.4',
            'name': file.stem,
            'datasets': [{
                'path': '0',
                'coordinate_transformations': [{
                    'type': 'scale',
                    'scale': scale
                }]
            }],
            'axes': [
                {'name': 't', 'type': 'time'},
                {'name': 'c', 'type': 'channel'},
                {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
            ]
        }]

        # Store scene names and channel information
        root.attrs['omero'] = {
            'id': 1,
            'name': file.stem,
            'version': '0.4',
            'scenes': lif_reader.scenes,
            'channels': [{'label': ch} for ch in lif_reader.channel_names]
        }

        # Verify the combined zarr
        opened_zarr = zarr.open(combined_path)
        print(f"Created combined zarr for {file.stem}")
        print(f"Combined shape: {combined_shape}")
        print(f"Zarr attributes: {dict(opened_zarr.attrs)}")

if __name__ == "__main__":
    if sys.argv[2] == "combine":
        #combine()
    elif sys.argv[2] == "zarr":
        main()
