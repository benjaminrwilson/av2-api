# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Sweep representation of lidar sensor data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import av2.structures.range_view as range_view
from av2.geometry.geometry import cart_to_sph
from av2.geometry.se3 import SE3
from av2.utils.constants import PI, TAU
from av2.utils.io import read_ego_SE3_sensor, read_feather
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayUShort


@dataclass
class Sweep:
    """Models a lidar sweep from a lidar sensor.

    A sweep refers to a set of points which were captured in a fixed interval [t,t+delta), where delta ~= (1/sensor_hz).
    Reference: https://en.wikipedia.org/wiki/Lidar

    NOTE: Argoverse 2 distributes sweeps which are from two, stacked Velodyne 32 beam sensors.
        These sensors each have different, overlapping fields-of-view.
        Both lidars have their own reference frame: up_lidar and down_lidar, respectively.
        We have egomotion-compensated the lidar sensor data to the egovehicle reference timestamp (`timestamp_ns`).

    Args:
        xyz: (N,3) Points in Cartesian space (x,y,z) in meters.
        intensity: (N,1) Intensity values in the interval [0,255] corresponding to each point.
        laser_number: (N,1) Laser numbers in the interval [0,63] corresponding to the beam which generated the point.
        offset_ns: (N,1) Nanosecond offsets _from_ the start of the sweep.
        timestamp_ns: Nanosecond timestamp _at_ the start of the sweep.
        ego_SE3_up_lidar: Pose of the up lidar in the egovehicle reference frame. Translation is in meters.
        ego_SE3_down_lidar: Pose of the down lidar in the egovehicle reference frame. Translation is in meters.
    """

    xyz: NDArrayFloat
    intensity: NDArrayByte
    laser_number: NDArrayByte
    offset_ns: NDArrayInt
    timestamp_ns: int
    ego_SE3_up_lidar: SE3
    ego_SE3_down_lidar: SE3

    def __len__(self) -> int:
        """Return the number of LiDAR returns in the aggregated sweep."""
        return len(self.xyz)

    def as_range_view(
        self,
        n_inclination_bins: int = 64,
        n_azimuth_bins: int = 1800,
        range_resolution: float = 0.005,
        offset_ns_resolution: float = 0.001,
    ) -> range_view.RangeView:
        """Convert a set of points in R^3 (x,y,z) to range image of shape (n_inclination_bins,n_azimuth_bins,range).

        Args:
            n_inclination_bins: Vertical resolution of the range image.
            n_azimuth_bins: Horizontal resolution of the range image.
            range_resolution: Size of each discrete range bin.
            offset_ns_resolution: Size of each discrete offset bin.

        Returns:
            The range image containing range, intensity, and nanosecond offset.
        """
        xyz_up_lidar = self.ego_SE3_up_lidar.inverse().transform_point_cloud(self.xyz)
        offset_ns = self.offset_ns

        sph = cart_to_sph(xyz_up_lidar)

        az = sph[..., 0]
        inc = sph[..., 1]
        rad = sph[..., 2]
        intensity = self.intensity

        perm: NDArrayInt = np.argsort(rad)
        inc = inc[perm]
        az, rad = az[perm], rad[perm]
        intensity = intensity[perm]
        offset_ns = offset_ns[perm]

        az += PI
        az *= n_azimuth_bins / TAU
        az_idx = az.astype(int)
        inc_idx = range_view.LASER_NUMBER_TO_ROW[self.laser_number][perm]

        inc_mask = np.greater_equal(inc_idx, 0) & np.less(inc_idx, n_inclination_bins)
        az_mask = np.greater_equal(az_idx, 0) & np.less(az_idx, n_azimuth_bins)

        mask = np.logical_and(inc_mask, az_mask)
        inc_idx = inc_idx[mask]
        az_idx = az_idx[mask]
        rad = rad[mask]
        offset_ns = offset_ns[mask]

        offset_ns = np.divide(offset_ns, offset_ns_resolution)
        shape = (n_inclination_bins, n_azimuth_bins, 1)

        rad = np.divide(rad, range_resolution)
        rad = np.round(rad)
        range_im: NDArrayUShort = np.full(shape, fill_value=range_view.RANGE_FILL_VALUE, dtype=np.uint16)
        range_im[inc_idx, az_idx, 0] = rad.astype(np.uint16)

        intensity_im: NDArrayByte = np.full(shape, fill_value=range_view.INTENSITY_FILL_VALUE, dtype=np.uint8)
        intensity_im[inc_idx, az_idx, 0] = intensity

        offset_im: NDArrayUShort = np.full(shape, fill_value=range_view.RANGE_FILL_VALUE, dtype=np.uint16)
        offset_im[inc_idx, az_idx, 0] = offset_ns

        return range_view.RangeView(
            range=range_im,
            intensity=intensity_im,
            offset_ns=offset_im,
            timestamp_ns=self.timestamp_ns,
            ego_SE3_up_lidar=self.ego_SE3_up_lidar,
            ego_SE3_down_lidar=self.ego_SE3_down_lidar,
            range_resolution=range_resolution,
            offset_ns_resolution=offset_ns_resolution,
        )

    @classmethod
    def from_feather(cls, lidar_feather_path: Path) -> Sweep:
        """Load a lidar sweep from a feather file.

        NOTE: The feather file is expected in AV2 format.
        NOTE: The sweep is in the _ego_ reference frame.

        The file should be a Apache Feather file and contain the following columns:
            x: Coordinate of each lidar return along the x-axis.
            y: Coordinate of each lidar return along the y-axis.
            z: Coordinate of each lidar return along the z-axis.
            intensity: Measure of radiant power per unit solid angle.
            laser_number: Laser which emitted the point return.
            offset_ns: Nanosecond delta from the sweep timestamp for the point return.

        Args:
            lidar_feather_path: Path to the lidar sweep feather file.

        Returns:
            Sweep object.
        """
        timestamp_ns = int(lidar_feather_path.stem)
        lidar = read_feather(lidar_feather_path)

        xyz = lidar.loc[:, ["x", "y", "z"]].to_numpy().astype(float)
        intensity = lidar.loc[:, ["intensity"]].to_numpy().squeeze()
        laser_number = lidar.loc[:, ["laser_number"]].to_numpy().squeeze()
        offset_ns = lidar.loc[:, ["offset_ns"]].to_numpy().squeeze()

        log_dir = lidar_feather_path.parent.parent.parent
        sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
        ego_SE3_up_lidar = sensor_name_to_pose["up_lidar"]
        ego_SE3_down_lidar = sensor_name_to_pose["down_lidar"]

        return cls(
            xyz=xyz,
            intensity=intensity,
            laser_number=laser_number,
            offset_ns=offset_ns,
            timestamp_ns=timestamp_ns,
            ego_SE3_up_lidar=ego_SE3_up_lidar,
            ego_SE3_down_lidar=ego_SE3_down_lidar,
        )
