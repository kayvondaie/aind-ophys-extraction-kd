import json
import logging
import os
import sys
from datetime import datetime as dt
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from typing import Optional, Tuple, Union

import caiman
import cv2
import h5py
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import skimage
import sparse
import suite2p
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_ophys_utils.array_utils import downsample_array
from aind_ophys_utils.summary_images import (max_corr_image, max_image,
                                             mean_image)
from aind_qcportal_schema.metric_value import CheckboxMetric, CurationMetric
from caiman.source_extraction.cnmf import cnmf, params
from cellpose.models import Cellpose
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.sparse import coo_matrix, hstack, linalg


class ExtractionSettings(BaseSettings, cli_parse_args=True):
    """Settings for extraction pipeline using pydantic-settings."""

    # Basic parameters
    input_dir: Path = Field(
        default=Path("../data/"),
        description="Directory containing the input data files.",
    )
    output_dir: Path = Field(
        default=Path("../results/"),
        description="Directory where output files will be saved.",
    )
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description="Directory for temporary files created during processing.",
    )

    # Cell detection parameters
    diameter: int = Field(
        default=0,
        description=(
            "Expected diameter of cells in pixels. "
            "If set to 0, CellPose will estimate the diameter from the data."
        ),
    )
    init: str = Field(
        default="sparsery",
        description=(
            "Initialization method for finding masks. Options: "
            "max/mean: Cellpose on max projection image divided by mean image; "
            "mean: Cellpose on mean image; "
            "enhanced_mean: Cellpose on enhanced mean image; "
            "max: Cellpose on maximum projection image; "
            "sourcery: Suite2p's functional mode without 'sparse_mode'; "
            "sparsery: Suite2p's functional mode with 'sparse_mode'; "
            "greedy_roi: CaImAn's functional 'greedy_roi' mode; "
            "corr_pnr: CaImAn's functional 'corr_pnr' mode."
        ),
    )
    functional_chan: int = Field(
        default=1,
        description="this channel is used to extract functional ROIs (1-based)",
    )
    threshold_scaling: float = Field(
        default=0.7,
        description="adjust the automatically determined threshold by this scalar multiplier",
    )
    max_overlap: float = Field(
        default=0.75,
        description="cells with more overlap than this get removed during triage, before refinement",
    )
    soma_crop: bool = Field(
        default=True,
        description="crop dendrites for cell classification stats like compactness",
    )
    allow_overlap: bool = Field(
        default=False,
        description="pixels that are overlapping are thrown out (False) or added to both ROIs (True)",
    )
    denoise: bool = Field(
        default=False,
        description=(
            "If True, applies denoising to the binned movie before cell detection."
        ),
    )
    cellprob_threshold: float = Field(
        default=0.0,
        description=(
            "Probability threshold for CellPose cell detection. "
            "Decrease this threshold if CellPose is not returning enough ROIs."
        ),
    )
    flow_threshold: float = Field(
        default=1.5,
        description=(
            "Flow threshold used by CellPose during cell detection. "
            "Increase this threshold if CellPose is not returning enough ROIs."
        ),
    )
    spatial_hp_cp: int = Field(
        default=0,
        description=(
            "Window size for spatial high-pass filtering of the image before CellPose "
            "detection. Set to 0 to disable filtering."
        ),
    )
    pretrained_model: str = Field(
        default="cyto",
        description=(
            "CellPose pretrained model to use. Common options: 'cyto' (standard model), "
            "'cyto2' (improved model), or path to a custom model file."
        ),
    )
    # Neuropil parameters
    neuropil: str = Field(
        default="mutualinfo",
        description=(
            "Method to estimate and subtract neuropil contamination, and whether to "
            "perform demixing. cnmf(-e) demix traces of overlapping ROIs via NMF, "
            "suite2p & mutualinfo do not. Options: "
            "'cnmf' (CaImAn standard: low-rank background of CNMF), "
            "'cnmf-e' (CaImAn for endoscopic data: ring model of CNMF-E ), "
            "'suite2p' (fixed r=0.7), "
            "'mutualinfo' (optimize r by minimizing mutual information)."
        ),
    )
    # CNMF parameters
    K: int = Field(
        default=5,
        description=(
            "Maximum number of components to be found per patch when using greedy_roi "
            "or corr_pnr initialization in CaImAn."
        ),
    )
    nb: int = Field(
        default=2,
        description=(
            "Number of background components if using CaImAn with neuropil=cnmf."
        ),
    )
    rf: Optional[int] = Field(
        default=40,
        description=(
            "Half-size of patches in pixels for CaImAn processing. If 0, the entire FOV "
            "is processed as a single patch. Larger patches are more memory-intensive."
        ),
    )
    stride: int = Field(
        default=18,
        description=(
            "Overlap between neighboring patches in pixels for CaImAn processing. "
            "Should be smaller than rf."
        ),
    )
    tsub: int = Field(
        default=2,
        description=(
            "Temporal downsampling factor during initialization phase in CaImAn. "
            "Higher values speed up processing but may miss transients."
        ),
    )
    ssub: int = Field(
        default=2,
        description=(
            "Spatial downsampling factor during initialization phase in CaImAn. "
            "Higher values speed up processing but may miss smaller cells."
        ),
    )
    ssub_B: int = Field(
        default=2,
        description=(
            "Additional spatial downsampling factor for background "
            "during CNMF-E processing."
        ),
    )
    merge_thr: float = Field(
        default=0.8,
        description=(
            "Trace correlation threshold for merging components in CaImAn. "
            "Components with correlation above this value will be merged."
        ),
    )

    # CORR_PNR parameters
    min_corr: float = Field(
        default=0.6,
        description=(
            "Minimum local correlation for a component to be considered in corr_pnr "
            "initialization. Higher values result in fewer, more reliable components."
        ),
    )
    min_pnr: float = Field(
        default=4,
        description=(
            "Minimum peak-to-noise ratio for a component to be considered in corr_pnr "
            "initialization. Higher values result in fewer, more reliable components."
        ),
    )

    # Component evaluation parameters
    snr_thr: float = Field(
        default=1.5,
        description=(
            "Signal-to-noise ratio threshold for component acceptance in CaImAn "
            "evaluation. Components below this value will be rejected."
        ),
    )
    rval_thr: float = Field(
        default=0.6,
        description=(
            "Spatial correlation threshold for component acceptance in CaImAn "
            "evaluation. Components below this value will be rejected."
        ),
    )
    cnn_thr: float = Field(
        default=0.9,
        description=(
            "CNN classifier threshold for component acceptance in CaImAn evaluation. "
            "Components below this value will be rejected. Set to 0 to disable "
            "CNN-based classification."
        ),
    )

    # Output options
    contour_video: bool = Field(
        default=False,
        description=(
            "If True, creates a video overlaying raw data, ROI activity, and residual "
            "with contours for visualization and quality assessment."
        ),
    )

    verbose: bool = Field(
        default=False, description="Enable verbose logging and debug information."
    )

    # Config for pydantic-settings
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="EXTRACTION_", case_sensitive=False, extra="ignore"
    )

    @field_validator("init", "neuropil")
    @classmethod
    def lowercase_str_fields(cls, v: str) -> str:
        """Convert string fields to lowercase"""
        return v.lower()

    @field_validator("rf")
    @classmethod
    def validate_rf(cls, v: int) -> Optional[int]:
        if v == 0:
            return None
        return v

    def validate_consistency(self) -> Optional[str]:
        """Validate command line arguments for consistency"""
        if self.neuropil == "cnmf" and self.init == "corr_pnr":
            # We'll log a warning but still update the parameters
            self.ssub = 1
            return (
                "'corr_pnr' initialization with neuropil model 'cnmf' does "
                "not support spatial downsampling. Setting ssub to 1"
            )

        if self.neuropil == "cnmf-e" and self.init == "greedy_roi":
            raise ValueError(
                "Can't use neuropil model 'cnmf-e' with 'greedy_roi' initialization"
            )

        if self.init in ("greedy_roi", "corr_pnr") and self.neuropil[:4] != "cnmf":
            raise ValueError(
                "Can't use Suite2p neuropil model with 'greedy_roi' or 'corr_pnr' initialization"
            )

        # For backward compatibility
        if self.init in ("1", "2", "3", "4"):
            self.init = ("max/mean", "mean", "enhanced_mean", "max")[int(self.init) - 1]

        return None  # No warning message

    def model_post_init(self, _) -> None:
        """Run validation after model initialization"""
        warning = self.validate_consistency()
        if warning:
            logging.warning(warning)


def get_metadata(input_dir: Path) -> Tuple[dict, dict, dict]:
    """Get the session and data description metadata from the input directory

    Parameters
    ----------
    input_dir: Path
        input directory

    Returns
    -------
    session: dict
        session metadata
    data_description: dict
        data description metadata
    subject: dict
        subject metadata
    """
    session_fp = next(input_dir.rglob("session.json"))
    with open(session_fp, "r") as j:
        session = json.load(j)
    data_des_fp = next(input_dir.rglob("data_description.json"))
    with open(data_des_fp, "r") as j:
        data_description = json.load(j)
    subject_fp = next(input_dir.rglob("subject.json"))
    with open(subject_fp, "r") as j:
        subject = json.load(j)

    return session, data_description, subject


def get_frame_rate(session: dict) -> float:
    """Attempt to pull frame rate from session.json
    Returns none if frame rate not in session.json

    Parameters
    ----------
    session: dict
        session metadata

    Returns
    -------
    frame_rate: float
        frame rate in Hz
    """
    frame_rate_hz = None
    for i in session.get("data_streams", ""):
        if i.get("ophys_fovs", ""):
            frame_rate_hz = i["ophys_fovs"][0]["frame_rate"]
            break
    if frame_rate_hz is None:
        raise ValueError("Frame rate not found in session.json")
    if isinstance(frame_rate_hz, str):
        frame_rate_hz = float(frame_rate_hz)
    return frame_rate_hz


def make_output_directory(output_dir: Path, experiment_id: str) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        experiment_id number

    Returns
    -------
    output_dir: str
        output directory
    """
    output_dir = output_dir / experiment_id / "extraction"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def write_data_process(
    metadata: dict,
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    unique_id: str,
    start_time: dt,
    end_time: dt,
) -> None:
    """Writes output metadata to plane data_process.json

    Parameters
    ----------
    metadata: dict
        parameters from suite2p motion correction
    input_fp: str
        path to raw movies
    output_fp: str
        path to motion corrected movies
    unique_id: str
        unique identifier for the session
    start_time: dt
        start time of the process
    end_time: dt
        end time of the process
    """
    if isinstance(input_fp, Path):
        input_fp = str(input_fp)
    if isinstance(output_fp, Path):
        output_fp = str(output_fp)
    data_proc = DataProcess(
        name=ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION,
        software_version=os.getenv("VERSION", ""),
        start_date_time=start_time.isoformat(),
        end_date_time=end_time.isoformat(),
        input_location=input_fp,
        output_location=output_fp,
        code_url=(os.getenv("REPO_URL", "")),
        parameters=metadata,
    )
    if isinstance(output_fp, str):
        output_dir = Path(output_fp).parent
    else:
        output_dir = output_fp.parent
    with open(output_dir / f"{unique_id}_extraction_data_process.json", "w") as f:
        json.dump(json.loads(data_proc.model_dump_json()), f, indent=4)


def write_qc_metrics(output_dir: Path, experiment_id: str, num_rois: int) -> None:
    """Write the QC metrics to a json file

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        unique plane id
    num_rois: int
        number of ROIs detected in this plane
    """

    # Build options and statuses
    options = []
    statuses = []

    options.append("Missing ROIs")
    statuses.append(Status.FAIL)

    for i in range(num_rois):
        options.append(f"ROI {i} invalid")
        statuses.append(Status.FAIL)

    # Define metric
    metric = QCMetric(
        name=f"{experiment_id} Detected ROIs",
        description="",
        reference=str(
            f"{experiment_id}/extraction/{experiment_id}_detected_ROIs_withIDs.png"
        ),
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=CheckboxMetric(value=[], options=options, status=statuses),
    )

    with open(output_dir / f"{experiment_id}_extraction_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


# Data Handling Functions
def create_virtual_dataset(
    h5_file: Path, frame_locations: list, frames_length: int, temp_dir: Path
) -> Path:
    """Creates a virtual dataset from a list of frame locations

    Parameters
    ----------
    h5_file: Path
        path to h5 file
    frame_locations: list
        list of frame locations
    frames_length: int
        sum of frame locations
    temp_dir: Path
        temporary directory for virtual dataset

    Returns
    -------
    h5_file: Path
        path to virtual dataset
    """
    with h5py.File(h5_file, "r") as f:
        data_shape = f["data"].shape
        dtype = f["data"].dtype
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(shape=(frames_length, *data_shape[1:]), dtype=dtype)
        start = 0
        for loc in frame_locations:
            layout[start : start + loc[1] - loc[0] + 1] = vsource[loc[0] : loc[1] + 1]
            start += loc[1] - loc[0] + 1
        h5_file = temp_dir / h5_file.name
        with h5py.File(h5_file, "w") as f:
            f.create_virtual_dataset("data", layout)

    return h5_file


def bergamo_segmentation(motion_corr_fp: Path, session: dict, temp_dir: Path) -> Path:
    """Creates a virtual dataset for Bergamo segmentation by filtering out photostimulation frames

    Parameters
    ----------
    motion_corr_fp: Path
        path to data directory
    session: dict
        session information
    temp_dir: Path
        temporary directory for virtual dataset
    Returns
    -------
    h5_file: Path
        path to motion corrected h5 file
    """
    motion_dir = motion_corr_fp.parent
    epoch_loc_fp = next(motion_dir.glob("epoch_locations.json"))
    with open(epoch_loc_fp, "r") as j:
        epoch_locations = json.load(j)
    valid_epoch_stems = [
        i["output_parameters"]["tiff_stem"]
        for i in session["stimulus_epochs"]
        if i["stimulus_name"] != "2p photostimulation"
    ]
    frame_locations = [epoch_locations[i] for i in valid_epoch_stems]
    frames_length = sum([(i[1] - i[0] + 1) for i in frame_locations])

    return create_virtual_dataset(
        motion_corr_fp, frame_locations, frames_length, temp_dir
    )


def create_chunk_vds(start: int, chunksize: int, input_fn: str, tmp_dir: str) -> str:
    """
    Create a Virtual Dataset (VDS) for a specific chunk of the input data.

    Parameters
    ----------
    start : int
        The starting index of the chunk in the input dataset.
    chunksize : int
        The number of rows in the chunk.
    input_fn : str
        Path to the input HDF5 file containing the source dataset.
    tmp_dir : str
        Directory where the temporary VDS file for the chunk will be created.

    Returns
    -------
    str
        The path to the created VDS file for the chunk.
    """
    with h5py.File(input_fn, "r") as fin:
        data = fin["data"]
        end = min(start + chunksize, data.shape[0])
        # Define the virtual layout for this chunk
        layout = h5py.VirtualLayout(
            shape=(end - start, *data.shape[1:]), dtype=data.dtype
        )
        vsource = h5py.VirtualSource(input_fn, "data", shape=data.shape)
        layout[:] = vsource[start:end]
        # Create a VDS file for this chunk
        vds_file = os.path.join(tmp_dir, f"chunk_{start}.h5")
        with h5py.File(vds_file, "w") as fout:
            fout.create_virtual_dataset("data", layout)
    return vds_file


def create_mmap_file(
    input_fn: str,
    unique_id: str,
    tmp_dir: str,
    chunksize: int = 500,
    n_chunks: int = 100,
) -> str:
    """
    Create a memory-mapped file from input data.

    Parameters
    ----------
    input_fn : str
        Path to the input HDF5 file containing the source dataset.
    unique_id : str
        Unique identifier for the output memory-mapped file.
    tmp_dir : str
        Directory where temporary chunk files or virtual datasets will be created.
    chunksize : int, optional
        The number of frames in each chunk (default is 500).
    n_chunks : int, optional
        The number of chunks to use when combining the memory-mapped file (default is 100).

    Returns
    -------
    str
        The path to the created memory-mapped file.
    """
    with h5py.File(input_fn, "r") as fin:
        data = fin["data"]
        if data.nbytes < 1e9:
            logging.info("Data is small, saving directly as a memory-mapped file.")
            fname_new = caiman.save_memmap(
                [str(input_fn)],
                var_name_hdf5="data",
                order="C",
                base_name=unique_id,
            )
        else:
            logging.info("Data is large, splitting into chunks.")
            with Pool() as pool:
                try:
                    chunkfiles = pool.starmap(
                        create_chunk_vds,
                        [
                            (start, chunksize, input_fn, tmp_dir)
                            for start in range(0, data.shape[0], chunksize)
                        ],
                    )
                    logging.info(f"Created {len(chunkfiles)} chunk files.")
                    fname_new = caiman.save_memmap(
                        chunkfiles,
                        var_name_hdf5="data",
                        order="C",
                        dview=pool,
                        base_name=unique_id,
                        n_chunks=n_chunks,
                    )
                    logging.info("Memory-mapped file created successfully.")
                finally:
                    logging.info("Cleaning up temporary chunk files.")
                    pool.map(os.remove, chunkfiles)
    return fname_new


# ROI Analysis Functions
def com(rois: Union[np.ndarray, sparse.COO]) -> np.ndarray:
    """Calculation of the center of mass for spatial components

    Parameters
    ----------
    rois : np.ndarray or sparse.COO tensor
        Tensor of Spatial components (K x height x width)

    Returns
    -------
    cm : np.ndarray
        center of mass for spatial components (K x 2)
    """
    d1, d2 = rois.shape[1:]
    Coor = np.array(
        list(map(np.ravel, np.meshgrid(np.arange(d2), np.arange(d1)))), dtype=rois.dtype
    )
    A = rois.reshape((rois.shape[0], d1 * d2)).tocsc()
    return (A / A.sum(axis=1)).dot(Coor.T)


def get_contours(
    rois: Union[np.ndarray, sparse.COO], thr: float = 0.2, thr_method: str = "max"
) -> list[dict]:
    """Gets contour of spatial components and returns their coordinates

    Parameters
    ----------
    rois : np.ndarray or sparse.COO tensor
        Tensor of Spatial components (K x height x width)
    thr : float between 0 and 1, optional
        threshold for computing contours, by default 0.2
    thr_method : str, optional
        Method of thresholding:
            'max' sets to zero pixels that have value less than a fraction of the max value
            'nrg' keeps the pixels that contribute up to a specified fraction of the energy

    Returns
    -------
    coordinates : list
        list of coordinates with center of mass and contour plot coordinates for each component
    """

    nr, dims = rois.shape[0], rois.shape[1:]
    d1, d2 = dims[:2]
    d = np.prod(dims)
    x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = com(rois)
    A = rois.T.reshape((d, nr)).tocsc()

    for i in range(nr):
        pars = dict()
        # we compute the cumulative sum of the energy of the Ath
        # component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i] : A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == "nrg":
            cumEn = np.cumsum(patch_data[indx] ** 2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                # we work with normalized values
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                # we put it in a similar matrix
                Bvec[A.indices[A.indptr[i] : A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != "max":
                logging.warning("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i] : A.indptr[i + 1]]] = (
                patch_data / patch_data.max()
            )

        Bmat = np.reshape(Bvec, dims, order="F")
        pars["coordinates"] = []
        # for each dimensions we draw the contour
        for B in Bmat if len(dims) == 3 else [Bmat]:
            vertices = skimage.measure.find_contours(B.T, thr)
            # this fix is necessary for having disjoint figures and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars["coordinates"] = v if len(dims) == 2 else (pars["coordinates"] + [v])
        pars["CoM"] = np.squeeze(cm[i, :])
        pars["neuron_id"] = i + 1
        coordinates.append(pars)
    return coordinates


def estimate_gSig(diameter: float, img: np.ndarray, fac: float = 2.35482) -> float:
    """Estimate Gaussian sigma for CaImAn based on cell diameter.

    Parameters
    ----------
    diameter : float
        Cell diameter in pixels; if 0, it will be automatically estimated with Cellpose.
    img : np.ndarray
        Mean image used for automatic diameter estimation if needed.
    fac : float
        Factor by which to divide Cellpose's diameter estimate.
        Based on jGCaMP data, a value between 2 and 2.5 is a good choice.
        Default is 2 sqrt(2 ln(2)), based on the FWHM of a Gaussian.

    Returns
    -------
    float
        Estimated Gaussian sigma.
    """
    if diameter == 0:
        diameter = Cellpose().sz.eval(img)[0]
        logger.info(
            f"'diameter' set to 0 — automatically estimated with Cellpose as {diameter:.3f}."
        )
    gSig = diameter / fac
    logger.info(f"Setting gSig to {gSig:.3f}.")
    return gSig


# Trace Processing Functions
def get_r_from_min_mi(
    raw_trace: np.ndarray,
    neuropil_trace: np.ndarray,
    resolution: float = 0.01,
    r_test_range: list[float] = [0, 2],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Get the r value that minimizes the mutual information between
    the corrected trace and the neuropil trace.

    Parameters
    ----------
    raw_trace : np.ndarray
        1D array of raw trace values.
    neuropil_trace : np.ndarray
        1D array of neuropil trace values.
    resolution : float
        Resolution of r values to test.
    r_test_range : list of float
        List of two floats representing the inclusive range of r values to test.

    Returns
    -------
    r_best : float
        The r value that minimizes the mutual information between
        the corrected trace and the neuropil trace.
    mi_iters : np.ndarray
        1D array of mutual information values for each r value tested.
    r_iters : np.ndarray
        1D array of r values tested.
    """
    r_iters = np.arange(r_test_range[0], r_test_range[1] + resolution, resolution)
    mi_iters = np.zeros(len(r_iters))
    neuropil_trace[np.isnan(neuropil_trace)] = 0
    raw_trace[np.isnan(raw_trace)] = 0
    for r_i, r_temp in enumerate(r_iters):
        Fc = raw_trace - r_temp * neuropil_trace
        mi_iters[r_i] = skimage.metrics.normalized_mutual_information(
            Fc, neuropil_trace
        )
    min_ind = np.argmin(mi_iters)
    r_best = r_iters[min_ind]
    return r_best, mi_iters, r_iters


def get_FC_from_r(
    raw_trace: np.ndarray, neuropil_trace: np.ndarray, min_r_count: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the corrected trace from the raw trace and neuropil trace using optimal r values.

    Parameters
    ----------
    raw_trace : np.ndarray
        2D array of raw trace values (ROIs x time).
    neuropil_trace : np.ndarray
        2D array of neuropil trace values (ROIs x time).
    min_r_count : int
        Minimum number of valid r values (< 1) required to calculate a mean r value.
        If fewer valid values are available, a default of 0.8 is used.

    Returns
    -------
    FCs : np.ndarray
        2D array of corrected traces for each ROI (ROIs x time).
    r_values : np.ndarray
        1D array of r values used for the correction.
    raw_r : np.ndarray
        1D array of r values that minimized the mutual information before thresholding.
    """
    r_values = np.zeros(raw_trace.shape[0])
    FCs = np.zeros_like(raw_trace)
    for roi in range(raw_trace.shape[0]):
        r_values[roi], _, _ = get_r_from_min_mi(raw_trace[roi], neuropil_trace[roi])
    mean_r = np.mean(r_values[r_values < 1])
    if len(np.where(r_values < 1)[0]) < min_r_count:
        mean_r = 0.8
    raw_r = r_values.copy()
    r_values[r_values >= 1] = mean_r
    for roi in range(raw_trace.shape[0]):
        FCs[roi] = raw_trace[roi] - r_values[roi] * neuropil_trace[roi]
    return FCs, r_values, raw_r


# CaImAn Functions
def build_CNMFParams(
    args: ExtractionSettings,
    ops: dict,
    cnmfe: bool,
    Ain: Optional[np.ndarray] = None,
    dims: Optional[tuple] = None,
) -> params.CNMFParams:
    """Build parameter dictionary for CaImAn extraction.

    Parameters
    ----------
    args : ExtractionSettings
        Command line arguments
    ops : dict
        Dictionary with summary images and other data
    cnmfe : bool
        Whether to use CNMF-E (True) or standard CNMF (False)
    Ain : scipy.sparse matrix, optional
        Initial spatial components (default: None)
    dims : tuple, optional
        Dimensions of data (needed for seed_method if Ain is provided)

    Returns
    -------
    params.CNMFParams
        Parameter dictionary for CaImAn
    """
    # Estimate Gaussian sigma based on cell diameter
    gSig = estimate_gSig(args.diameter, ops["meanImg"])

    # Base parameters (common to both initialization methods)
    params_dict = {
        "p": 1,
        "nb": 0 if cnmfe else args.nb,
        "only_init": cnmfe,
        "gSig": (gSig, gSig),
        "gSiz": (int(round(gSig * (4 if cnmfe else 2) + 1)),) * 2,
        "ssub_B": args.ssub_B,
        "normalize_init": not cnmfe,
        "center_psf": cnmfe,
        "ring_size_factor": 1.5 if cnmfe else None,
        "min_SNR": args.snr_thr,
        "rval_thr": args.rval_thr,
        "use_cnn": args.cnn_thr > 0,
        "min_cnn_thr": args.cnn_thr,
    }

    # Additional parameters specific to initialization method
    if Ain is None:
        # For initial ROI detection (greedy_roi or corr_pnr)
        params_dict.update(
            {
                "K": args.K,
                "method_init": args.init,
                "rf": args.rf,
                "stride": args.stride,
                "ssub": args.ssub,
                "tsub": args.tsub,
                "min_corr": args.min_corr,
                "min_pnr": args.min_pnr,
                "merge_thr": args.merge_thr,
            }
        )
    else:
        # For refining Suite2p ROIs
        params_dict.update(
            {
                "K": None,
                "method_init": "corr_pnr" if cnmfe else "greedy_roi",
                "ssub": 1,
                "tsub": 1,
                "min_corr": 0,
                "min_pnr": 0,
                "merge_thr": 1,
                "init_iter": 1,
            }
        )

        # Add seed method if dims is provided
        if Ain is not None and dims is not None:
            params_dict["seed_method"] = caiman.base.rois.com(Ain, *dims)

    return params.CNMFParams(params_dict=params_dict)


def run_caiman_extraction(
    input_fn: Union[str, Path],
    unique_id: str,
    args: ExtractionSettings,
    ops: dict,
    Ain: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CaImAn to extract neural activity traces.

    Parameters
    ----------
    input_fn : str or Path
        Path to the input HDF5 file containing the source dataset.
    unique_id : str
        Unique identifier for the session.
    args : ExtractionSettings
        Command line arguments.
    ops : dict
        Dictionary with summary images and other data.
    Ain : scipy.sparse matrix, optional
        Initial spatial components (default: None).
        If None, will perform initial ROI detection.
        If provided, will refine the provided ROIs.
    n_jobs : int, optional
        Number of parallel processes to use.

    Returns
    -------
    tuple
        Tuple containing (traces_corrected, traces_neuropil, traces_roi, data, coords, iscell)
    """
    logger.info(f"running CaImAn v{caiman.__version__}")
    # Determine if using CNMF-E
    cnmfe = args.neuropil == "cnmf-e"
    # Create mmap file
    fname_new = create_mmap_file(input_fn, unique_id, str(args.tmp_dir))
    # Load the file
    Yr, dims, T = caiman.load_memmap(fname_new)
    movie = np.reshape(Yr.T, [T] + list(dims), order="F")
    # Create parameter object
    opts = build_CNMFParams(args, ops, cnmfe, Ain, dims)
    # Create Ain to use
    Ain_processed = None if cnmfe or Ain is None else (Ain > 0).toarray()
    # Run CNMF
    with Pool(n_jobs) as pool:
        cnm = cnmf.CNMF(
            n_processes=pool._processes,
            dview=pool,
            params=opts,
            Ain=Ain_processed,
        )
        cnm.fit(movie)
        # For standard CNMF, refit to improve results
        if not cnmfe and Ain is None:
            cnm = cnm.refit(movie, dview=pool)
        # Make sure dims are set and gSig is integer for component evaluation
        cnm.estimates.dims = dims
        cnm.params.init["gSig"] = tuple(map(int, cnm.params.init["gSig"]))
        # Evaluate components
        cnm.estimates.evaluate_components(movie, cnm.params, dview=pool)
    # Return formatted output
    return format_caiman_output(cnm.estimates, cnmfe, Yr)


def format_caiman_output(
    e: caiman.source_extraction.cnmf.estimates.Estimates, cnmfe: bool, Yr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Format the output from CaImAn's CNMF for standardized extraction results.

    Parameters
    ----------
    e : caiman.source_extraction.cnmf.estimates.Estimates
        The estimates object from CaImAn containing spatial and temporal components
    cnmfe : bool
        Flag indicating whether CNMF-E (True) or standard CNMF (False) was used
    Yr : np.ndarray
        The data in a flattened format (pixels x time)

    Returns
    -------
    traces_corrected : np.ndarray
        Array of corrected fluorescence traces (ROIs x time)
    traces_neuropil : np.ndarray
        Array of neuropil traces (ROIs x time)
    traces_roi : np.ndarray
        Array of raw fluorescence traces including neuropil (ROIs x time)
    data : np.ndarray
        Values for ROI spatial footprints in sparse format
    coords : np.ndarray
        Coordinates for ROI spatial footprints in sparse format (3 x N)
    iscell : np.ndarray
        Array indicating for each component (ROI) whether it's a cell (1) or not (0)
    """
    assert np.allclose(linalg.norm(e.A, 2, 0), 1)
    traces_corrected = (e.C + e.YrA).astype("f4")
    if cnmfe:
        Atb0 = e.A.T.dot(e.b0)[:, None]
        traces_corrected += Atb0
        ssub_B = np.round(np.sqrt(Yr.shape[0] / e.W.shape[0])).astype(int)
        if ssub_B == 1:
            AtW = e.A.T.dot(e.W)
            traces_neuropil = (
                Atb0 + AtW.dot(Yr) - AtW.dot(e.A).dot(e.C) - AtW.dot(e.b0)[:, None]
            ).astype("f4")
        else:
            ds_mat = caiman.source_extraction.cnmf.utilities.decimation_matrix(
                e.dims, ssub_B
            )
            Ads = ds_mat.dot(e.A)
            b0ds = ds_mat.dot(e.b0)
            AtW = Ads.T.dot(e.W)
            traces_neuropil = (
                Atb0
                + ssub_B**2
                * (
                    AtW.dot(ds_mat).dot(Yr)
                    - AtW.dot(Ads).dot(e.C)
                    - AtW.dot(b0ds)[:, None]
                )
            ).astype("f4")
    else:
        traces_neuropil = e.A.T.dot(e.b).dot(e.f).astype("f4")
        traces_corrected += (
            0.8 * traces_neuropil
        )  # TODO: check factor on groundtruth data
    traces_roi = (e.C + e.YrA + traces_neuropil).astype("f4")
    # convert ROIs to sparse COO 3D-tensor (https://sparse.pydata.org/en/stable/construct.html)
    data = []
    coords = []
    for i in range(e.A.shape[1]):
        roi = coo_matrix(e.A[:, i].reshape(e.dims, order="F").toarray(), dtype="f4")
        data.append(roi.data)
        coords.append(
            np.array([i * np.ones(len(roi.data)), roi.row, roi.col], dtype="i2")
        )
    if len(data):
        data = np.concatenate(data)
        coords = np.hstack(coords)
    # maybe TODO: save background
    iscell = np.zeros((e.A.shape[1], 2), dtype="f4")
    iscell[e.idx_components, 0] = 1
    iscell[:, 1] = e.cnn_preds if hasattr(e, "cnn_preds") else np.nan
    return traces_corrected, traces_neuropil, traces_roi, data, coords, iscell


# Visualization Functions
def save_summary_images_with_rois(
    output_dir: Path,
    unique_id: str,
    rois: sparse.COO,
    iscell: np.ndarray,
    ops: dict,
    corr_img: np.ndarray,
) -> None:
    """Save summary images with ROI contours

    Parameters
    ----------
    output_dir : Path
        Directory where summary images will be saved
    unique_id : str
        Unique identifier for the output files
    rois : sparse.COO
        Tensor of spatial components (K x height x width)
    iscell : np.ndarray
        Array of shape (K, 2) indicating whether each component is a cell
    ops : dict
        Dictionary containing summary images
    corr_img : np.ndarray
        Correlation image
    """
    cm = com(rois)
    coordinates = get_contours(rois)
    dims = rois.shape[1:]
    # Create plots
    x_size = 17 * max(dims[1] / dims[0], 0.4)
    fix, ax = plt.subplots(1, 3, figsize=(x_size, 6))
    lw = min(512 / max(*dims), 3)
    for i, img in enumerate((ops["meanImg"], ops["max_proj"], corr_img)):
        vmin, vmax = np.nanpercentile(img, (1, 99))
        ax[i].imshow(img, interpolation=None, cmap="gray", vmin=vmin, vmax=vmax)
        for c, good in zip(coordinates, iscell[:, 0]):
            ax[i].plot(*c["coordinates"].T, c="orange" if good else "r", lw=lw)
        ax[i].axis("off")
        ax[i].set_title(
            ("mean image", "max image", "correlation image")[i],
            fontsize=min(24, 2.4 + 2 * x_size),
        )
    plt.tight_layout(pad=0.1)
    plt.savefig(
        output_dir / f"{unique_id}_detected_ROIs.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    # Add IDs and save another version
    for i in (0, 1, 2):
        for k in range(rois.shape[0]):
            ax[i].text(
                *cm[k], str(k), color="orange" if iscell[k, 0] else "r", fontsize=8 * lw
            )
    plt.savefig(
        output_dir / f"{unique_id}_detected_ROIs_withIDs.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )


def contour_video(
    output_path: str,
    data: Union[h5py.Dataset, np.ndarray],
    rois: Union[sparse.COO, np.ndarray],
    traces: np.ndarray,
    downscale: int = 10,
    fs: float = 30,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.9975,
    only_raw: bool = False,
    n_jobs: Optional[int] = (
        None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)
    ),
    bitrate: str = "0",
    crf: int = 20,
    cpu_used: int = 4,
) -> None:
    """Create a video contours using vp9 codec via imageio-ffmpeg

    Parameters
    ----------
    output_path : str
        Desired output path for encoded video
    data : h5py.Dataset or numpy.ndarray
        Video to be encoded
    rois : np.ndarray or sparse.COO tensor
        Tensor of spatial components (K x height x width)
    traces: np.ndarray
        Tensor of temporal components (K x T)
    downscale : int = 10
        Decimation factor
    fs : float
        Desired frame rate for encoded video
    lower_quantile : float
        Lower cutoff value supplied to `np.quantile()` for normalization
    upper_quantile : float
        Upper cutoff value supplied to `np.quantile()` for normalization
    only_raw : bool, optional
        Produce video of raw data only, i.e. no reconstruction and residual
    n_jobs : int, optional
        The number of jobs to run in parallel.
    bitrate : str, optional
        Desired bitrate of output, by default "0". The default *MUST*
        be zero in order to encode in constant quality mode. Other values
        will result in constrained quality mode.
    crf : int, optional
        Desired perceptual quality of output, by default 20. Value can
        be from 0 - 63. Lower values mean better quality (but bigger video
        sizes).
    cpu_used : int, optional
        Sets how efficient the compression will be, by default 4. Values can
        be between 0 and 5. Higher values increase encoding speed at the
        expense of having some impact on quality and rate control accuracy.
    """
    dims = data.shape[1:]
    # create image of countours
    img_contours = np.zeros(dims + (3,), np.uint8)
    rgb = (255, 127, 14)
    for m in rois:
        if isinstance(m, sparse.COO):
            m = m.todense()
        ret, thresh = cv2.threshold(
            (m > m.max() / 10).astype(np.uint8), 0, 1, cv2.THRESH_BINARY
        )
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
        for contour in contours:
            cv2.drawContours(img_contours, contour, -1, rgb, max(max(dims) // 200, 1))
    # assemble movie tiles
    mov = downsample_array(data, downscale, 1, n_jobs=n_jobs)
    minmov, maxmov = np.nanquantile(
        mov[:: max(1, len(mov) // 100)], (lower_quantile, upper_quantile)
    )

    def scale(m):
        return np.array(
            ThreadPool(n_jobs).map(
                lambda frame: np.clip(
                    255 * (frame - minmov) / (maxmov - minmov), 0, 255
                ).astype(np.uint8),
                m,
            )
        )

    if only_raw:
        mov = scale(mov)
    else:
        img_contours = np.repeat(img_contours[..., None], 3, 0).reshape(
            dims[0], 3 * dims[1], -1
        )
        reconstructed = np.tensordot(
            downsample_array(traces.T, downscale, 1, n_jobs=n_jobs).astype("f4"),
            rois,
            1,
        )
        residual = scale(mov - reconstructed)
        mov = scale(mov)
        reconstructed = scale(reconstructed)
        mov = np.concatenate([mov, reconstructed, residual], 2)
        del reconstructed
        del residual
    # create canvas with labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    magnify = max(500 // dims[0], 1)
    h, w = dims[0] * magnify, dims[1] * magnify
    canvas_size = (
        int(np.ceil(h * 1.08 / 16)) * 16,
        int(np.ceil((w if only_raw else 3 * w) / 16)) * 16,
    )
    pad = (canvas_size[1] - 3 * w) // 2
    canvas = np.zeros(canvas_size + (3,), np.uint8)
    for i in range(1 if only_raw else 3):
        text = ("Original", "ROI Activity", "Residual")[i]
        fontscale = min(h / 600, w / 190)
        textsize = cv2.getTextSize(text, font, fontscale, max(h // 200, 1))[0]
        cv2.putText(
            canvas,
            text,
            (int(w * (0.49, 1.5, 2.51)[i] + pad - textsize[0] / 2), h // 25),
            font,
            fontscale,
            (255, 255, 255),
            max(h // 200, 1),
            cv2.LINE_4,
        )
    # create writer object
    writer = imageio_ffmpeg.write_frames(
        output_path,
        # ffmpeg expects video shape in terms of: (width, height)
        canvas_size[::-1],
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        codec="libvpx-vp9",
        fps=fs,
        bitrate=bitrate,
        output_params=[
            "-crf",
            str(crf),
            "-row-mt",
            "1",
            "-cpu-used",
            str(cpu_used),
        ],
    )
    writer.send(None)  # Seed ffmpeg-imageio writer generator
    # overlay image of contours and write each frame
    if magnify > 1:
        img_contours = cv2.resize(img_contours, (0, 0), fx=magnify, fy=magnify)
    is_contours = img_contours != 0
    for frame in mov:
        if magnify > 1:
            frame = cv2.resize(frame, (0, 0), fx=magnify, fy=magnify)
        frame = np.repeat(frame[..., None], 3, 2)
        frame[is_contours] = img_contours[is_contours]
        canvas[-h:, -(w if only_raw else 3 * w) :] = frame
        writer.send(canvas)
    writer.close()


if __name__ == "__main__":
    start_time = dt.now()
    # Parse command-line arguments
    args = ExtractionSettings()

    # Set the log level and name the logger
    logger = logging.getLogger(
        "Source extraction using a combination of Cellpose, Suite2p, and CaImAn"
    )
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # set env variables for CaImAn
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["CAIMAN_TEMP"] = str(args.tmp_dir)

    output_dir = args.output_dir.resolve()
    input_dir = args.input_dir.resolve()
    if next(input_dir.glob("output"), ""):
        sys.exit()
    tmp_dir = args.tmp_dir.resolve()
    try:
        session, data_description, subject = get_metadata(input_dir)
    except StopIteration:
        session, data_description, subject = {}, {}, {}
    subject_id = subject.get("subject_id", "")
    name = data_description.get("name", "")
    if next(input_dir.rglob("*decrosstalk.h5"), ""):
        input_fn = next(input_dir.rglob("*decrosstalk.h5"))
    else:
        input_fn = next(input_dir.rglob("*registered.h5"))
    unique_id = input_fn.parent.parent.name
    if session != {} and "Bergamo" in session["rig_id"]:
        motion_corrected_fn = bergamo_segmentation(input_fn, session, temp_dir=tmp_dir)
    else:
        motion_corrected_fn = input_fn
    frame_rate = get_frame_rate(session)
    output_dir = make_output_directory(output_dir, unique_id)

    n_jobs = tmp if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)

    if args.init in ("greedy_roi", "corr_pnr"):
        # Run CaImAn
        # ==========
        with h5py.File(str(motion_corrected_fn), "r") as open_vid:
            dims = open_vid["data"][0].shape
            ops = {
                "meanImg": mean_image(open_vid["data"]),
                "max_proj": max_image(open_vid["data"]),
            }
        (
            traces_corrected,
            traces_neuropil,
            traces_roi,
            data,
            coords,
            iscell,
        ) = run_caiman_extraction(
            input_fn, unique_id, args, ops, Ain=None, n_jobs=n_jobs
        )
        neuropil_coords, keys = [], []
        input_args = vars(args)

    else:
        # Run Cellpose via Suite2p to get ROI seeds
        # =========================================
        # Set suite2p args.
        suite2p_args = suite2p.default_ops()
        # Overwrite the parameters for suite2p that are exposed
        suite2p_args["diameter"] = args.diameter
        if args.diameter == 0 and args.init == "sourcery":
            with h5py.File(str(motion_corrected_fn), "r") as open_vid:
                suite2p_args["diameter"] = round(
                    Cellpose().sz.eval(mean_image(open_vid["data"]))[0]
                )
            logger.info(
                "'diameter' set to 0 — automatically estimated with Cellpose "
                f"as {suite2p_args['diameter']:.0f}."
            )
        suite2p_args["anatomical_only"] = {
            "max/mean": 1,
            "mean": 2,
            "enhanced_mean": 3,
            "max": 4,
        }.get(args.init, 0)
        suite2p_args["cellprob_threshold"] = args.cellprob_threshold
        suite2p_args["flow_threshold"] = args.flow_threshold
        suite2p_args["spatial_hp_cp"] = args.spatial_hp_cp
        suite2p_args["pretrained_model"] = args.pretrained_model
        suite2p_args["denoise"] = args.denoise
        suite2p_args["save_path0"] = str(tmp_dir)
        suite2p_args["functional_chan"] = args.functional_chan
        suite2p_args["threshold_scaling"] = 0.7  # hardcoded to match local pipeline; ignores CLI/env override
        suite2p_args["smooth_sigma"] = 0.5
        suite2p_args["max_overlap"] = args.max_overlap
        suite2p_args["soma_crop"] = args.soma_crop
        suite2p_args["allow_overlap"] = args.allow_overlap
        # Here we overwrite the parameters for suite2p that will not change in our
        # processing pipeline. These are parameters that are not exposed to
        # minimize code length. Those are not set to default.
        suite2p_args["sparse_mode"] = args.init == "sparsery"
        suite2p_args["nonrigid"] = False
        suite2p_args["h5py"] = str(motion_corrected_fn)
        suite2p_args["data_path"] = []
        suite2p_args["roidetect"] = True
        suite2p_args["do_registration"] = 0
        suite2p_args["spikedetect"] = False
        suite2p_args["fs"] = 20  # hardcoded to match local pipeline (actual metadata rate was ~58 Hz)
        suite2p_args["neuropil_extract"] = True
        # determine nbinned from bin_duration and fs
        # The duration of time (in seconds) that
        suite2p_args["bin_duration"] = 3.7
        # should be considered 1 bin for Suite2P ROI detection purposes. Requires
        # a valid value for 'fs' in order to derive an
        # nbinned Suite2P value. This allows consistent temporal downsampling
        # across movies with different lengths and/or frame rates.
        with h5py.File(suite2p_args["h5py"], "r") as f:
            nframes = f["data"].shape[0]
        bin_size = suite2p_args["bin_duration"] * suite2p_args["fs"]
        suite2p_args["nbinned"] = int(nframes / bin_size)
        logger.info(
            f"Movie has {nframes} frames collected at "
            f"{suite2p_args['fs']} Hz. "
            "To get a bin duration of "
            f"{suite2p_args['bin_duration']} "
            f"seconds, setting nbinned to "
            f"{suite2p_args['nbinned']}."
        )

        logger.info(f"running Suite2P v{suite2p.version}")
        try:
            input_args = {**vars(args), **suite2p_args}
            suite2p.run_s2p(suite2p_args)
        except IndexError:  # raised when no ROIs found
            pass

        # load in the rois from the stat file and movie path for shape
        with h5py.File(str(motion_corrected_fn), "r") as open_vid:
            dims = open_vid["data"][0].shape
        if len(list(tmp_dir.rglob("stat.npy"))):
            suite2p_stat_path = str(next(tmp_dir.rglob("stat.npy")))
            suite2p_stats = np.load(suite2p_stat_path, allow_pickle=True)
            if args.neuropil in ("mutualinfo", "suite2p"):
                # Run Suite2p to extract traces
                # =============================
                if session is not None and "Bergamo" in session["rig_id"]:
                    # extract signals for all frames, not just those used for cell detection
                    (
                        stat,
                        traces_roi,
                        traces_neuropil,
                        _,
                        _,
                    ) = suite2p.extraction.extraction_wrapper(
                        suite2p_stats, h5py.File(input_fn)["data"], ops=suite2p_args
                    )
                else:  # all frames have already been used for detection as well as extraction
                    suite2p_f_path = str(next(tmp_dir.rglob("F.npy")))
                    suite2p_fneu_path = str(next(tmp_dir.rglob("Fneu.npy")))
                    traces_roi = np.load(suite2p_f_path, allow_pickle=True)
                    traces_neuropil = np.load(suite2p_fneu_path, allow_pickle=True)
                iscell = np.load(str(next(tmp_dir.rglob("iscell.npy"))))
                if args.neuropil == "suite2p":
                    traces_corrected = (
                        traces_roi - suite2p_args["neucoeff"] * traces_neuropil
                    )
                    r_values = suite2p_args["neucoeff"] * np.ones(traces_roi.shape[0])
                else:
                    traces_corrected, r_values, raw_r = get_FC_from_r(
                        traces_roi, traces_neuropil
                    )
                # convert ROIs to sparse COO 3D-tensor
                data = []
                coords = []
                neuropil_coords = []
                for i, roi in enumerate(suite2p_stats):
                    data.append(roi["lam"])
                    coords.append(
                        np.array(
                            [i * np.ones(len(roi["lam"])), roi["ypix"], roi["xpix"]],
                            dtype=np.int16,
                        )
                    )
                    neuropil_coords.append(
                        np.array(
                            [
                                i * np.ones(len(roi["neuropil_mask"])),
                                roi["neuropil_mask"] // dims[1],
                                roi["neuropil_mask"] % dims[1],
                            ],
                            dtype=np.int16,
                        )
                    )
                keys = list(suite2p_stats[0].keys())
                for k in ("ypix", "xpix", "lam", "neuropil_mask"):
                    keys.remove(k)
                stat = {}
                for k in keys:
                    stat[k] = [s[k] for s in suite2p_stats]
                data = np.concatenate(data)
                coords = np.hstack(coords)
                neuropil_coords = np.hstack(neuropil_coords)
                stat["soma_crop"] = np.concatenate(stat["soma_crop"])
                stat["overlap"] = np.concatenate(stat["overlap"])

            else:
                # Run CaImAn to update ROIs and extract traces
                # ============================================
                Ain = hstack(
                    [
                        coo_matrix((roi["lam"], (roi["ypix"], roi["xpix"])), shape=dims)
                        .reshape((-1, 1), order="F")
                        .tocsc()
                        for roi in suite2p_stats
                    ]
                )
                ops_path = str(next(tmp_dir.rglob("ops.npy"), ""))
                ops = np.load(ops_path, allow_pickle=True)[()]
                (
                    traces_corrected,
                    traces_neuropil,
                    traces_roi,
                    data,
                    coords,
                    iscell,
                ) = run_caiman_extraction(
                    input_fn, unique_id, args, ops, Ain=Ain, n_jobs=n_jobs
                )
                neuropil_coords, keys = [], []

        else:  # no ROIs found
            traces_roi, traces_neuropil, traces_corrected = [
                np.empty((0, nframes), dtype=np.float32)
            ] * 3
            r_values, data, coords, neuropil_coords, iscell = [[]] * 5
            if args.neuropil == "mutualinfo":
                raw_r = []
            keys = []

    # write output files
    cellpose_path = str(next(tmp_dir.rglob("cellpose.npz"), ""))
    ops_path = str(next(tmp_dir.rglob("ops.npy"), ""))
    with h5py.File(output_dir / f"{unique_id}_extraction.h5", "w") as f:
        # traces
        f.create_dataset("traces/corrected", data=traces_corrected, compression="gzip")
        f.create_dataset("traces/neuropil", data=traces_neuropil, compression="gzip")
        f.create_dataset("traces/roi", data=traces_roi, compression="gzip")
        if args.neuropil in ("mutualinfo", "suite2p"):
            f.create_dataset("traces/neuropil_rcoef", data=r_values)
            if args.neuropil == "mutualinfo":
                # We save the raw r values if we are not using the suite2p neuropil.
                # This is useful for debugging purposes.
                f.create_dataset("traces/raw_neuropil_rcoef_mutualinfo", data=raw_r)
        for k in keys:
            dtype = np.array(stat[k]).dtype
            if dtype != "bool":
                dtype = "i2" if np.issubdtype(dtype, np.integer) else "f4"
            if k in ("skew", "std"):
                f.create_dataset(f"traces/{k}", data=stat[k], dtype=dtype)
            # ROIs
            else:
                f.create_dataset(f"rois/{k}", data=stat[k], dtype=dtype)
        f.create_dataset("rois/coords", data=coords, compression="gzip")
        f.create_dataset("rois/data", data=data, compression="gzip")
        shape = np.array([len(traces_roi), *dims], dtype=np.int16)
        f.create_dataset("rois/shape", data=shape)  # neurons x height x width
        if len(neuropil_coords) > 0:
            f.create_dataset(
                "rois/neuropil_coords", data=neuropil_coords, compression="gzip"
            )
        # cellpose
        if cellpose_path:
            with np.load(cellpose_path) as cp:
                for k in cp.keys():
                    f.create_dataset(f"cellpose/{k}", data=cp[k], compression="gzip")
        else:
            logging.warning("No cellpose output found.")

        # classifier
        f.create_dataset("iscell", data=iscell, dtype="f4")
        # summary images
        if ops_path:
            ops = np.load(ops_path, allow_pickle=True)[()]
        f.create_dataset("meanImg", data=ops["meanImg"], compression="gzip")
        f.create_dataset("maxImg", data=ops["max_proj"], compression="gzip")

    write_data_process(
        input_args,
        input_fn,
        output_dir / f"{unique_id}_extraction.h5",
        unique_id,
        start_time,
        dt.now(),
    )

    # plot contours of detected ROIs over a selection of summary images
    rois = sparse.COO(coords, data, shape)
    with h5py.File(str(motion_corrected_fn), "r") as f:
        corr_img = max_corr_image(f["data"])
    save_summary_images_with_rois(output_dir, unique_id, rois, iscell, ops, corr_img)
    import shutil
    suite2p_dirs = list(tmp_dir.rglob("plane0"))
    if suite2p_dirs:
        for f in Path(suite2p_dirs[0]).iterdir():
            if f.suffix in ('.npy',):
                shutil.copy(str(f), str(output_dir / f.name))
                print(f"DEBUG: copied {f.name}")

    # Per-epoch suite2p output split (BCI / spont_pre / spont_post / ...).
    # The bergamo_segmentation step concatenates non-photostim epochs in
    # `valid_epoch_stems` order before extraction, so we just slice the
    # combined F/Fneu/spks by frame range from epoch_locations.json. ROI-level
    # files (stat, iscell, ops, redcell) are per-ROI not per-frame -- copy as-is.
    if suite2p_dirs and session != {} and "Bergamo" in session.get("rig_id", ""):
        motion_dir = input_fn.parent
        epoch_loc_fp = next(motion_dir.glob("epoch_locations.json"), None)
        if epoch_loc_fp is not None:
            with open(epoch_loc_fp, "r") as j:
                epoch_locations = json.load(j)
            valid_epoch_stems = [
                i["output_parameters"]["tiff_stem"]
                for i in session["stimulus_epochs"]
                if i["stimulus_name"] != "2p photostimulation"
            ]
            s2p_plane0 = Path(suite2p_dirs[0])
            F_all = np.load(s2p_plane0 / "F.npy") if (s2p_plane0 / "F.npy").exists() else None
            Fneu_all = np.load(s2p_plane0 / "Fneu.npy") if (s2p_plane0 / "Fneu.npy").exists() else None
            spks_all = np.load(s2p_plane0 / "spks.npy") if (s2p_plane0 / "spks.npy").exists() else None
            roi_files = [n for n in ("stat.npy", "iscell.npy", "ops.npy", "redcell.npy")
                         if (s2p_plane0 / n).exists()]
            offset = 0
            for stem in valid_epoch_stems:
                loc = epoch_locations.get(stem)
                if loc is None:
                    print(f"DEBUG: no epoch_locations entry for stem '{stem}', skipping")
                    continue
                length = loc[1] - loc[0] + 1
                epoch_dir = output_dir / f"suite2p_{stem}" / "plane0"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                if F_all is not None:
                    np.save(epoch_dir / "F.npy", F_all[:, offset:offset + length])
                if Fneu_all is not None:
                    np.save(epoch_dir / "Fneu.npy", Fneu_all[:, offset:offset + length])
                if spks_all is not None:
                    np.save(epoch_dir / "spks.npy", spks_all[:, offset:offset + length])
                for rf in roi_files:
                    shutil.copy(str(s2p_plane0 / rf), str(epoch_dir / rf))
                print(f"DEBUG: wrote suite2p_{stem}/plane0 (frames {offset}:{offset + length}, {length} frames)")
                offset += length

    # create a video overlaid wit ROI contours
    if args.contour_video:
        with h5py.File(str(motion_corrected_fn), "r") as f:
            contour_video(
                output_dir / f"{unique_id}_ROI_contours_overlay.webm",
                f["data"],
                rois,
                traces_corrected,
                fs=frame_rate,
            )

    write_qc_metrics(output_dir, unique_id, num_rois=rois.shape[0])
