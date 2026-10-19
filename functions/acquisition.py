"""
functions/acquisition.py

Orchestration routines that combine the general-purpose hardware
functions (laser, camera, ...) into full acquisition sequences.
Never opens files itself: frames go back to the experiment script, or
into a core/file_io.py writer the script hands in.

`run_timelapse` below is the legacy, pycromanager-based orchestration
routine. `start_continuous_timelapse` is the pymmcore-plus one, driving
the acquisition through pymmcore-plus's MDA engine.
"""
import threading
import time
from datetime import datetime

import numpy as np

from functions.camera import set_exposure, snap_image
from functions.laser import laser_off, laser_on, set_laser_power


def run_timelapse(core, config, logger, num_frames, interval_s, exposure_ms, laser_power_mW):
    """
    (legacy, pycromanager) Acquire a simple single-channel, single-plane
    timelapse: laser on, snap `num_frames` images spaced `interval_s`
    apart, laser off.

    Returns
    -------
    frames : np.ndarray, shape (T, Y, X)
    metadata : dict
    """
    set_exposure(core, exposure_ms)
    set_laser_power(core, config, laser_power_mW)
    laser_on(core, config)

    frames = []
    timestamps = []

    try:
        for i in range(num_frames):
            t0 = time.time()

            frame = snap_image(core)
            frames.append(frame)
            timestamps.append(datetime.now().isoformat())
            logger.info(f"Frame {i + 1}/{num_frames} acquired.")

            if i < num_frames - 1:
                elapsed = time.time() - t0
                time.sleep(max(0.0, interval_s - elapsed))
    finally:
        laser_off(core, config)

    metadata = {
        "num_frames": num_frames,
        "interval_s": interval_s,
        "exposure_ms": exposure_ms,
        "laser_power_mW": laser_power_mW,
        "timestamps": timestamps,
        "slm_mask_note": "flat/neutral phase mask",
    }
    return np.stack(frames, axis=0), metadata


def sampling_rate_hz(exposure_ms):
    """
    Frame rate of a continuous timelapse if the camera is exposure-limited:
    1 / exposure. It's an upper bound - at short exposures the sensor
    readout (and so the ROI size) can limit the rate instead; the measured
    rate is reported after each run.
    """
    return 1000.0 / float(exposure_ms)


class TimelapseRecorder:
    """
    pymmcore-plus MDA output handler for `start_continuous_timelapse`:
    hands each frame to `writer.write(t, frame)` and keeps its camera
    timestamp, then on sequenceFinished switches the laser off and calls
    `on_finished(summary)`. Called on pymmcore-plus's relay thread, which
    the MDA waits on before it finishes, so no frame is lost.
    """

    def __init__(self, mmc, config, writer, num_frames, on_finished=None):
        self.mmc, self.config, self.writer = mmc, config, writer
        self.num_frames = num_frames
        self.on_finished = on_finished
        self.times_ms = []
        self.latest = None          # (t, frame), for a live preview
        self.started = None
        self.done = threading.Event()
        self.summary = None
        self._finish_lock = threading.Lock()

    def sequenceStarted(self, *args):
        self.started = time.perf_counter()

    def frameReady(self, img, event, meta):
        t = event.index.get("t", 0)
        self.writer.write(t, img)
        cam = (meta or {}).get("camera_metadata", {}) if isinstance(meta, dict) else {}
        stamp = cam.get("ElapsedTime-ms", (meta or {}).get("runner_time_ms"))
        self.times_ms.append(float(stamp) if stamp is not None else None)
        self.latest = (t, img)

    def sequenceFinished(self, *args):
        # Runs once: from the MDA's sequenceFinished signal, or from
        # `finish_if_needed` if that signal never got here (another listener
        # raising in it stops psygnal from calling the rest).
        with self._finish_lock:
            if self.done.is_set() or self.summary is not None:
                return
            self.summary = {}
        from functions.mmcore_laser import laser_off_mmcore

        laser_off_mmcore(self.mmc, self.config)
        n = len(self.times_ms)
        stamps = [s for s in self.times_ms if s is not None]
        measured = None
        if len(stamps) > 1 and stamps[-1] > stamps[0]:
            measured = (len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1000.0)
        self.summary = {
            "frames_requested": self.num_frames,
            "frames_acquired": n,
            "completed": n == self.num_frames,
            "elapsed_s": time.perf_counter() - self.started if self.started else None,
            "measured_sampling_rate_hz": measured,
            "frame_times_ms": self.times_ms,
        }
        try:
            if self.on_finished is not None:
                self.on_finished(self.summary)
        finally:
            self.done.set()
            _ACTIVE.discard(self)

    def finish_if_needed(self, mda_thread=None):
        """Wait for the MDA (its thread, if given) and make sure the run is finished."""
        if mda_thread is not None:
            mda_thread.join()
        if not self.done.wait(timeout=5.0):
            self.sequenceFinished()


# The MDA runner only holds output handlers weakly; a non-blocking run's
# recorder is kept alive here until it finishes.
_ACTIVE = set()


def start_continuous_timelapse(mmc, config, logger, writer, num_frames, exposure_ms,
                               laser_power_mW, block=True, on_finished=None):
    """
    Continuous timelapse at the current position: `num_frames` frames
    with no wait between them, laser on at `laser_power_mW` for the run.
    The MDA engine gets a plain list of time-point events, which it runs
    as one hardware-sequenced camera acquisition - no stage moves, no
    channel groups, no pixel-size calibration needed. (A list rather than
    an MDASequence also makes napari-micromanager show the frames in a
    single 'preview' layer, instead of mirroring the whole run into a
    temporary copy on disk.) Frames go to `writer` as they arrive (see
    TimelapseRecorder).

    block=True waits for the end; block=False returns at once (e.g. from
    a GUI) and `on_finished(summary)` is called when the run ends, also
    if it's stopped with `mmc.mda.cancel()` - the laser is switched off
    either way. Returns the TimelapseRecorder.
    """
    from useq import MDAEvent

    from functions.mmcore_laser import laser_off_mmcore, laser_on_mmcore, set_laser_power_mmcore

    if mmc.mda.is_running():
        raise RuntimeError("An acquisition is already running.")
    if mmc.isSequenceRunning():   # e.g. live view: the camera can't do both
        mmc.stopSequenceAcquisition()

    recorder = TimelapseRecorder(mmc, config, writer, num_frames, on_finished)
    _ACTIVE.add(recorder)
    mmc.setExposure(exposure_ms)
    set_laser_power_mmcore(mmc, config, laser_power_mW)
    laser_on_mmcore(mmc, config)
    logger.info(f"Continuous timelapse: {num_frames} frames, {exposure_ms} ms exposure "
                f"(<= {sampling_rate_hz(exposure_ms):.2f} Hz), laser {laser_power_mW} mW")
    events = [MDAEvent(index={"t": t}) for t in range(num_frames)]
    try:
        thread = mmc.run_mda(events, output=recorder, block=block)
    except Exception:
        laser_off_mmcore(mmc, config)
        _ACTIVE.discard(recorder)
        raise
    if block:
        recorder.finish_if_needed()
    else:
        threading.Thread(target=recorder.finish_if_needed, args=(thread,), daemon=True).start()
    return recorder
