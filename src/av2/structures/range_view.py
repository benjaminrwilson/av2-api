# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Range view representation of lidar sensor data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Tuple

import cv2
import numpy as np

import av2.structures.sweep as sweep
from av2.geometry.geometry import sph_to_cart
from av2.geometry.se3 import SE3
from av2.utils.constants import MAX_USHORT, PI, TAU
from av2.utils.io import read_ego_SE3_sensor
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat, NDArrayInt, NDArrayUShort

# fmt: off

# Maps laser number to range image row.
# Sorts the point cloud in descending inclination in spherical coordinates.
# Row zero corresponds to the first row of the range image.
LASER_NUMBER_TO_ROW: Final[NDArrayByte] = np.array([7, 41, 21, 35, 2, 33, 14, 27, 23, 31, 25, 18, 29, 37, 10, 4, 55, 62, 47, 43, 51, 58, 52, 48, 46, 54, 39, 57, 50, 60, 44, 63, 56, 22, 42, 28, 61, 30, 49, 36, 40, 32, 38, 45, 34, 26, 53, 59, 8, 1, 16, 20, 12, 5, 11, 15, 17, 9, 24, 6, 13, 3, 19, 0])
# fmt: on

# fmt: off

# Maps row to inclination of the laser.
ROW_TO_INC: Final[NDArrayFloat] = np.array([0.41812001611011573, 0.26457514795070913, 0.26208795734366563, 0.18909000630458322, 0.18007074128359762, 0.1462558842245967, 0.11874857301469961, 0.12189681074641004, 0.09955363078224237, 0.0853052293079634, 0.08112580571454298, 0.07392851436235781, 0.06254742125254634, 0.05685574222950682, 0.05794408655318252, 0.05124068883387316, 0.046060149860919344, 0.04066502461896573, 0.04053778783656782, 0.03501080766190992, 0.029439897208134915, 0.028936465807600977, 0.023877853162038346, 0.02310943524551832, 0.01812177428858862, 0.017302295408157246, 0.012844749395314244, 0.011490358366276028, 0.00782056376575424, 0.005656685576958256, 0.002382729791565715, -0.00019622207673795114, -0.003528617934719381, -0.005995994778373084, -0.009767180037466551, -0.011806553793009194, -0.015838053262276688, -0.0175955693680621, -0.02194923601050123, -0.023390341652695355, -0.02783744612522273, -0.029210560027228327, -0.03360991329484061, -0.034998650953189284, -0.040804912824534245, -0.04505687435161117, -0.04661753031970549, -0.052423738646802845, -0.0582273632083877, -0.06239116598301768, -0.064088948765293, -0.06989627112156303, -0.08152564946728344, -0.08589732484204834, -0.09313616561149299, -0.10739518011275508, -0.12811242988404253, -0.12667634571040295, -0.15444266551287367, -0.18960506205698932, -0.19749506331691888, -0.2733453168422842, -0.2729201742931131, -0.42910305882537964])
# fmt: on

# Assume 200m lidar range.
# Compute bucket resolution by dividing by maximum number of buckets.
RANGE_RESOLUTION: Final[float] = 200 / MAX_USHORT
OFFSET_NS_RESOLUTION: Final[float] = 1e3 / MAX_USHORT

RANGE_FILL_VALUE: Final[int] = np.iinfo(np.uint16).max
INTENSITY_FILL_VALUE: Final[int] = np.iinfo(np.uint8).max


@dataclass
class RangeView:
    """Models a sweep as a dense range image.

    Args:
        range_img: (H,W,1) Image representing the range in spherical lidar coordinates.
        xyz_img: (H,W,1) Image containing the (x,y,z) Cartesian coordinates.
        intensity_img: (H,W,1) Image representing the lidar intensity.
        offset_ns_img: (H,W,1) Image containing the nanosecond offsets _from_ the start of the sweep.
        tov_ns: (1,) Nanosecond timestamp _at_ the start of the sweep.
        ego_SE3_up_lidar: Pose of the up lidar in the egovehicle reference frame. Translation component is in meters.
        ego_SE3_down_lidar: Pose of the down lidar in the egovehicle reference frame. Translation component is in meters.
        range_resolution: Size of each discrete range bin.
        offset_ns_resolution: Size of each discrete offset bin.
    """

    range_img: NDArrayUShort
    xyz_img: NDArrayFloat
    intensity_img: NDArrayByte
    offset_ns_img: NDArrayUShort
    timestamp_ns: int
    ego_SE3_up_lidar: SE3
    ego_SE3_down_lidar: SE3
    range_resolution: float
    offset_ns_resolution: float

    def as_sweep(self) -> sweep.Sweep:
        """Converts a range image of shape (n_inclination_bins,n_azimuth_bins,range) to a set of points in R^3 (x,y,z).

        Returns:
            Sweep view of the point cloud.
        """
        n_inclination_bins, n_azimuth_bins = self.range_img.shape[:2]

        out: Tuple[NDArrayBool, ...] = np.nonzero(self.range_img != MAX_USHORT)
        inc_idx: NDArrayInt = out[0].astype(int)
        az_idx: NDArrayInt = out[1].astype(int)

        intensity = self.intensity_img[inc_idx, az_idx]
        offset_ns: NDArrayUShort = self.offset_ns_img[inc_idx, az_idx]

        rad = self.range_img[inc_idx, az_idx] * self.range_resolution
        inc = ROW_TO_INC[inc_idx]

        # Map azimuth bins to real-valued azimuth values.
        az: NDArrayFloat = az_idx * (TAU / n_azimuth_bins)
        az -= PI

        sph: NDArrayFloat = np.stack((az, inc, rad), axis=-1)

        xyz: NDArrayFloat = sph_to_cart(sph)
        xyz = self.ego_SE3_up_lidar.transform_point_cloud(xyz)
        laser_number: NDArrayByte = inc_idx.astype(np.uint8)

        return sweep.Sweep(
            xyz=xyz,
            intensity=intensity,
            laser_number=laser_number,
            offset_ns=offset_ns.astype(int),
            timestamp_ns=self.timestamp_ns,
            ego_SE3_up_lidar=self.ego_SE3_up_lidar,
            ego_SE3_down_lidar=self.ego_SE3_down_lidar,
        )

    @classmethod
    def from_png(cls, path: Path) -> RangeView:
        """Load a range view from a set of pngs.

        Expected directory structure:
            {tov_ns}/
                range.png
                intensity.png
                offset.png

        Args:
            path: Path to the sweep timestamp folder.

        Returns:
            The range view of the point cloud.
        """
        range_path = path / "range.png"
        xyz_path = path / "xyz.png"
        intensity_path = path / "intensity.png"
        offset_path = path / "offset_ns.png"

        range_img: NDArrayUShort = cv2.imread(str(range_path), cv2.IMREAD_ANYDEPTH)[..., None]
        xyz_img: NDArrayFloat = cv2.imread(str(xyz_path), cv2.IMREAD_ANYDEPTH)
        intensity_img: NDArrayByte = cv2.imread(str(intensity_path), cv2.IMREAD_ANYDEPTH)[..., None]
        offset_ns_img: NDArrayUShort = cv2.imread(str(offset_path), cv2.IMREAD_ANYDEPTH)[..., None]
        timestamp_ns = int(path.stem)

        ego_SE3_sensor = read_ego_SE3_sensor(Path(path).parent.parent.parent)
        ego_SE3_up_lidar = ego_SE3_sensor["up_lidar"]
        ego_SE3_down_lidar = ego_SE3_sensor["down_lidar"]

        return cls(
            range_img=range_img,
            xyz_img=xyz_img,
            intensity_img=intensity_img,
            offset_ns_img=offset_ns_img,
            timestamp_ns=timestamp_ns,
            ego_SE3_up_lidar=ego_SE3_up_lidar,
            ego_SE3_down_lidar=ego_SE3_down_lidar,
            range_resolution=RANGE_RESOLUTION,
            offset_ns_resolution=OFFSET_NS_RESOLUTION,
        )

    def write_png(self, path: Path) -> None:
        """Write a range view to a set of pngs.

        Expected directory structure:
            {tov_ns}/
                range.png
                intensity.png
                offset.png

        Args:
            path: Path to the sweep timestamp folder.
        """

        dst = Path(path) / str(self.timestamp_ns)
        dst.mkdir(parents=True, exist_ok=True)

        dst_range = dst / "range.png"
        dst_intensity = dst / "intensity.png"
        dst_offset_ns = dst / "offset_ns.png"

        cv2.imwrite(str(dst_range), self.range_img)
        cv2.imwrite(str(dst_intensity), self.intensity_img)
        cv2.imwrite(str(dst_offset_ns), self.offset_ns_img)
