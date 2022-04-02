# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Detection utilities for the Argoverse detection leaderboard.

Accepts detections (in Argoverse ground truth format) and ground truth labels
for computing evaluation metrics for 3d object detection. We have five different,
metrics: mAP, ATE, ASE, AOE, and CDS. A true positive for mAP is defined as the
highest confidence prediction within a specified Euclidean distance threshold
from a bird's-eye view. We prefer these metrics instead of IoU due to the
increased interpretability of the error modes in a set of detections.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from av2.evaluation.detection.constants import (
    MAX_NORMALIZED_ASE,
    MAX_SCALE_ERROR,
    MAX_YAW_RAD_ERROR,
    MIN_AP,
    MIN_CDS,
    AffinityType,
    CompetitionCategories,
    DistanceType,
    FilterMetricType,
    InterpType,
    TruePositiveErrorNames,
)
from av2.geometry.geometry import mat_to_xyz, quat_to_mat, wrap_angles
from av2.geometry.iou import iou_3d_axis_aligned
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.structures.cuboid import CuboidList
from av2.utils.constants import EPS
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayObject

logger = logging.getLogger(__name__)

CUBOID_COLS: Final[List[str]] = ["tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"]


@dataclass(frozen=True)
class DetectionCfg:
    """Instantiates a DetectionCfg object for configuring a evaluation.

    Args:
        affinity_thresholds_m: Affinity thresholds for determining a true positive (in meters).
        affinity_type: Type of affinity function to be used for calculating average precision.
        categories: Detection classes for evaluation.
        eval_only_roi_instances: Only use dets and ground truth that lie within region of interest during eval.
        filter_metric: Detection metric to use for filtering of both detections and ground truth annotations.
        max_range_m: Max distance (under a specific metric in meters) for a detection or ground truth cuboid to be
            considered for evaluation.
        num_recall_samples: Number of recall points to sample uniformly in [0, 1].
        splits: Tuple of split names to evaluate.
        tp_threshold_m: Center distance threshold for the true positive metrics (in meters).
    """

    affinity_thresholds_m: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    affinity_type: AffinityType = AffinityType.CENTER
    categories: Tuple[str, ...] = tuple(x.value for x in CompetitionCategories)
    dataset_dir: Optional[Path] = None
    eval_only_roi_instances: bool = True
    filter_metric: FilterMetricType = FilterMetricType.EUCLIDEAN
    max_num_dts_per_category: int = 100
    max_range_m: float = 200.0
    num_recall_samples: int = 101
    split: str = "val"
    tp_threshold_m: float = 2.0

    @property
    def metrics_defaults(self) -> Tuple[float, ...]:
        """Return the evaluation summary default values."""
        return (
            MIN_AP,
            self.tp_threshold_m,
            MAX_NORMALIZED_ASE,
            MAX_YAW_RAD_ERROR,
            MIN_CDS,
        )

    @property
    def tp_normalization_terms(self) -> Tuple[float, ...]:
        """Return the normalization constants for ATE, ASE, and AOE."""
        return (
            self.tp_threshold_m,
            MAX_SCALE_ERROR,
            MAX_YAW_RAD_ERROR,
        )


def accumulate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
    avm: Optional[ArgoverseStaticMap] = None,
    city_SE3_ego: Optional[SE3] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Accumulate the true / false positives (boolean flags) and true positive errors for each class.

    Args:
        dts: (N,len(AnnotationColumns)) Detections of shape. Must contain all of the columns in AnnotationColumns.
        gts: (M,len(AnnotationColumns) + 1) Ground truth labels. Must contain all of the columns in AnnotationColumns
            and the `num_interior_pts` column.
        cfg: Detection configuration.
        avm: Argoverse static map for the log.
        city_SE3_ego: Egovehicle pose in the city reference frame.

    Returns:
        The detection and ground truth cuboids augmented with assigment and evaluation fields.
    """
    # Sort the detections by score in _descending_ order.
    scores: NDArrayFloat = dts["score"].to_numpy()
    permutation: NDArrayInt = np.argsort(-scores)

    dts_npy: NDArrayFloat = dts.loc[:, CUBOID_COLS].to_numpy()[permutation]
    gts_npy: NDArrayFloat = gts.loc[:, CUBOID_COLS].to_numpy()

    num_interior_pts: NDArrayFloat = gts["num_interior_pts"].to_numpy()
    is_evaluated_dts = compute_evaluated_dts_mask(dts_npy, cfg)
    is_evaluated_gts = compute_evaluated_gts_mask(gts_npy, num_interior_pts, cfg)

    N, M = len(dts), len(gts)
    dt_results: NDArrayFloat = np.zeros((N, 8))
    gt_results: NDArrayFloat = np.zeros((M, 5))
    if is_evaluated_dts.sum() == 0 or is_evaluated_gts.sum() == 0:
        return dt_results, gt_results

    dts_npy = dts_npy[is_evaluated_dts]
    gts_npy = gts_npy[is_evaluated_gts]

    # Compute true positives through assigning detections and ground truths.
    dts_assignments, gts_assignments = assign(dts_npy, gts_npy, cfg)
    dt_results[is_evaluated_dts, :-1] = dts_assignments
    gt_results[is_evaluated_gts, :-1] = gts_assignments

    # `is_evaluated` boolean flag is always the last column of the array.
    dt_results[is_evaluated_dts, -1] = True
    gt_results[is_evaluated_gts, -1] = True

    # Permute the detections according to the original ordering.
    outputs: Tuple[NDArrayInt, NDArrayInt] = np.unique(permutation, return_index=True)
    _, inverse_permutation = outputs
    dt_results = dt_results[inverse_permutation]
    return dt_results, gt_results


def assign(dts: NDArrayFloat, gts: NDArrayFloat, cfg: DetectionCfg) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Attempt assignment of each detection to a ground truth label.

    Args:
        dts: (N,23) Detections of shape. Must contain all columns in AnnotationColumns and the
            additional columns: (is_evaluated, *cfg.affinity_thresholds_m, *TruePositiveErrorNames).
        gts: (M,23) Ground truth labels. Must contain all columns in AnnotationColumns and the
            additional columns: (is_evaluated, *cfg.affinity_thresholds_m, *TruePositiveErrorNames).
        cfg: Detection configuration.

    Returns:
        The (N,K+S) confusion table containing the true and false positives augmented with the true positive errors
            where K is the number of thresholds and S is the number of true positive error names.
    """
    affinity_matrix = compute_affinity_matrix(dts, gts, cfg.affinity_type)

    # Get the GT label for each max-affinity GT label, detection pair.
    idx_gts = affinity_matrix.argmax(axis=1)[None]

    # The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
    # We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
    # The following line grabs the max affinity for each detection to a ground truth label.
    affinities: NDArrayFloat = np.take_along_axis(affinity_matrix.transpose(), idx_gts, axis=0)[0]  # type: ignore

    # Find the indices of the _first_ detection assigned to each GT.
    assignments: Tuple[NDArrayInt, NDArrayInt] = np.unique(idx_gts, return_index=True)  # type: ignore

    idx_gts, idx_dts = assignments
    K = len(cfg.affinity_thresholds_m)
    dts_table: NDArrayFloat = np.zeros((len(dts), K + 3))
    dts_table[:, 4:] = cfg.metrics_defaults[1:4]
    gts_table: NDArrayFloat = np.zeros((len(gts), K))
    for i, threshold_m in enumerate(cfg.affinity_thresholds_m):
        is_tp: NDArrayBool = affinities[idx_dts] > -threshold_m

        dts_table[idx_dts[is_tp], i] = True
        gts_table[idx_gts, i] = True

        if threshold_m != cfg.tp_threshold_m:
            continue  # Skip if threshold isn't the true positive threshold.
        if not np.any(is_tp):
            continue  # Skip if no true positives exist.

        idx_tps_dts: NDArrayInt = idx_dts[is_tp]
        idx_tps_gts: NDArrayInt = idx_gts[is_tp]

        tps_dts = dts[idx_tps_dts]
        tps_gts = gts[idx_tps_gts]

        translation_errors = distance(tps_dts[:, :3], tps_gts[:, :3], DistanceType.TRANSLATION)
        scale_errors = distance(tps_dts[:, 3:6], tps_gts[:, 3:6], DistanceType.SCALE)
        orientation_errors = distance(tps_dts[:, 6:10], tps_gts[:, 6:10], DistanceType.ORIENTATION)

        dts_table[idx_tps_dts, 4:] = np.stack((translation_errors, scale_errors, orientation_errors), axis=-1)
    return dts_table, gts_table


def interpolate_precision(precision: NDArrayFloat, interpolation_method: InterpType = InterpType.ALL) -> NDArrayFloat:
    r"""Interpolate the precision at each sampled recall.

    This function smooths the precision-recall curve according to the method introduced in Pascal
    VOC:

    Mathematically written as:
        $$p_{\text{interp}}(r) = \max_{\tilde{r}: \tilde{r} \geq r} p(\tilde{r})$$

    See equation 2 in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.6629&rep=rep1&type=pdf
        for more information.

    Args:
        precision: Precision at all recall levels (N,).
        interpolation_method: Accumulation method.

    Returns:
        (N,) The interpolated precision at all sampled recall levels.

    Raises:
        NotImplementedError: If the interpolation method is not implemented.
    """
    precision_interpolated: NDArrayFloat
    if interpolation_method == InterpType.ALL:
        precision_interpolated = np.maximum.accumulate(precision[::-1])[::-1]
    else:
        raise NotImplementedError("This interpolation method is not implemented!")
    return precision_interpolated


def compute_affinity_matrix(dts: NDArrayFloat, gts: NDArrayFloat, metric: AffinityType) -> NDArrayFloat:
    """Calculate the affinity matrix between detections and ground truth annotations.

    Args:
        dts: (N,) Detections.
        gts: (M,) Ground truth annotations.
        metric: Affinity metric type.

    Returns:
        The affinity scores between detections and ground truth annotations (N,M).

    Raises:
        NotImplementedError: If the affinity metric is not implemented.
    """
    if metric == AffinityType.CENTER:
        dts_xy_m = dts[:, :2]
        gts_xy_m = gts[:, :2]
        affinities: NDArrayFloat = -cdist(dts_xy_m, gts_xy_m)
    else:
        raise NotImplementedError("This affinity metric is not implemented!")
    return affinities


def compute_average_precision(
    tps: NDArrayBool, recall_interpolated: NDArrayFloat, num_gts: int
) -> Tuple[float, NDArrayFloat]:
    """Compute precision and recall, interpolated over N fixed recall points.

    Args:
        tps: True positive detections (ranked by confidence).
        recall_interpolated: Interpolated recall values.
        num_gts: Number of annotations of this class.

    Returns:
        The average precision and interpolated precision values.
    """
    cum_tps: NDArrayInt = np.cumsum(tps)
    cum_fps: NDArrayInt = np.cumsum(~tps)
    cum_fns: NDArrayInt = num_gts - cum_tps

    # Compute precision.
    precision: NDArrayFloat = cum_tps / (cum_tps + cum_fps + EPS)

    # Compute recall.
    recall: NDArrayFloat = cum_tps / (cum_tps + cum_fns)

    # Interpolate precision -- VOC-style.
    precision = interpolate_precision(precision)

    # Evaluate precision at different recalls.
    precision_interpolated: NDArrayFloat = np.interp(recall_interpolated, recall, precision, right=0)  # type: ignore

    average_precision: float = np.mean(precision_interpolated)
    return average_precision, precision_interpolated


def distance(dts: NDArrayFloat, gts: NDArrayFloat, metric: DistanceType) -> NDArrayFloat:
    """Distance functions between detections and ground truth.

    Args:
        dts: (N,D) Detections where D is the number of attributes.
        gts: (N,D) Ground truth labels where D is the number of attributes.
        metric: Distance function type.

    Returns:
        (N,) Distance between the detections and ground truth under the specified metric.

    Raises:
        NotImplementedError: If the distance type is not supported.
    """
    if metric == DistanceType.TRANSLATION:
        translation_errors: NDArrayFloat = np.linalg.norm(dts - gts, axis=1)  # type: ignore
        return translation_errors
    elif metric == DistanceType.SCALE:
        scale_errors: NDArrayFloat = 1 - iou_3d_axis_aligned(dts, gts)
        return scale_errors
    elif metric == DistanceType.ORIENTATION:
        yaws_dts: NDArrayFloat = mat_to_xyz(quat_to_mat(dts))[..., 2]
        yaws_gts: NDArrayFloat = mat_to_xyz(quat_to_mat(gts))[..., 2]
        orientation_errors = wrap_angles(yaws_dts - yaws_gts)
        return orientation_errors
    else:
        raise NotImplementedError("This distance metric is not implemented!")


def compute_objects_in_roi_mask(
    cuboids_dataframe: pd.DataFrame, city_SE3_ego: SE3, avm: ArgoverseStaticMap
) -> NDArrayBool:
    """Compute the evaluated cuboids mask based off whether _any_ of their vertices fall into the ROI.

    Args:
        cuboids_dataframe: Dataframes containing cuboids.
        city_SE3_ego: Egovehicle pose in the city reference frame.
        avm: Argoverse map object.

    Returns:
        The boolean mask indicating which cuboids will be evaluated.
    """
    cuboids_dataframe = cuboids_dataframe.sort_values("timestamp_ns").reset_index(drop=True)

    cuboid_list_ego = CuboidList.from_dataframe(cuboids_dataframe)
    cuboid_list_city = cuboid_list_ego.transform(city_SE3_ego)
    cuboid_list_vertices_m = cuboid_list_city.vertices_m

    is_within_roi = avm.get_raster_layer_points_boolean(
        cuboid_list_vertices_m.reshape(-1, 3)[..., :2], RasterLayerType.ROI
    )
    is_within_roi = is_within_roi.reshape(-1, 8)
    is_within_roi = is_within_roi.any(axis=1)
    return is_within_roi


def compute_evaluated_dts_mask(
    dts: NDArrayFloat,
    cfg: DetectionCfg,
) -> NDArrayBool:
    """Compute the evaluated cuboids mask.

    Valid cuboids meet _two_ conditions:
        1. The cuboid's centroid (x,y,z) must lie within the maximum range in the detection configuration.
        2. The cuboid must have at _least_ one point in its interior.

    Args:
        dts: Dataframes containing cuboids.
        cfg: Detection configuration object.

    Returns:
        The boolean mask indicating which cuboids will be evaluated.
    """
    norm: NDArrayFloat = np.linalg.norm(dts[:, :3], axis=1)  # type: ignore
    is_within_radius: NDArrayBool = norm < cfg.max_range_m
    is_evaluated: NDArrayBool = is_within_radius
    is_evaluated[cfg.max_num_dts_per_category :] = False  # Limit the number of detections.
    return is_evaluated


def compute_evaluated_gts_mask(
    gts: NDArrayFloat,
    num_interior_pts: NDArrayFloat,
    cfg: DetectionCfg,
) -> NDArrayBool:
    """Compute the evaluated cuboids mask.

    Valid cuboids meet _two_ conditions:
        1. The cuboid's centroid (x,y,z) must lie within the maximum range in the detection configuration.
        2. The cuboid must have at _least_ one point in its interior.

    Args:
        gts: Dataframes containing ground truth cuboids.
        cfg: Detection configuration object.

    Returns:
        The boolean mask indicating which cuboids will be evaluated.
    """
    norm: NDArrayFloat = np.linalg.norm(gts[:, :3], axis=1)  # type: ignore
    is_within_radius: NDArrayBool = norm < cfg.max_range_m
    is_valid_num_points: NDArrayBool = num_interior_pts > 0
    is_evaluated: NDArrayBool = is_within_radius & is_valid_num_points
    return is_evaluated
