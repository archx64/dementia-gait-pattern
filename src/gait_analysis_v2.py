"""
GaitAnalyzer v2 — improvements over v1:

  1. Walking-direction PCA  — step length and step width are projected onto the
     actual walking axis (computed from heel trajectory), not assumed to be Z and X.

  2. Floor-collision event detection  — threshold crossings on the height signal
     replace find_peaks, mirroring a game-engine collider.  Heel strike = heel
     crosses below (floor + 5 cm); toe-off = toe crosses above (floor + 5 cm).
     Sub-frame linear interpolation gives precise crossing times and contact
     positions, which also fixes consecutive-contact step_width computation.

  3. Tighter event-detection filter  — a dedicated 4 Hz low-pass on heel/toe
     height before threshold detection suppresses residual keypoint jitter.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.utils_floor_align import (
    FPS_ANALYSIS,
    SUBJECT_NAME,
    ROUND,
    INFO,
    OUTPUT_CSV,
    DAY,
    MONTH,
    P_NO,
)

from src.foot_point_mle import ground_plane_from_alignment, estimate_stance_point_mle


class GaitAnalyzer:
    def __init__(
        self, csv_path, fps, height_axis="y", start_frame=None, end_frame=None
    ):
        """
        Parameters
        ----------
        csv_path     : path to floor-aligned skeleton CSV from pose_estimation.py
        fps          : capture frame rate
        height_axis  : column suffix for the vertical axis ("y")
        start_frame  : first frame to include (None = beginning)
        end_frame    : last frame to include  (None = end)
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.height_axis = height_axis.lower()

        print(INFO + f"Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)

        if start_frame is not None:
            self.df = self.df[self.df["frame_idx"] >= start_frame]
            print(INFO + f"Cropped start to frame >= {start_frame}")
        if end_frame is not None:
            self.df = self.df[self.df["frame_idx"] <= end_frame]
            print(INFO + f"Cropped end to frame <= {end_frame}")

        self.df = self.df.reset_index(drop=True)

        self.map = {
            "L_Heel_X": "j19_x",
            "L_Heel_Y": "j19_y",
            "L_Heel_Z": "j19_z",
            "R_Heel_X": "j22_x",
            "R_Heel_Y": "j22_y",
            "R_Heel_Z": "j22_z",
            "L_Toe_X": "j17_x",
            "L_Toe_Y": "j17_y",
            "L_Toe_Z": "j17_z",
            "R_Toe_X": "j20_x",
            "R_Toe_Y": "j20_y",
            "R_Toe_Z": "j20_z",
        }

        head_y = self.df["j0_y"].dropna().mean()
        heel_y = pd.concat([self.df["j19_y"], self.df["j22_y"]]).dropna().mean()
        if head_y > heel_y:
            self.up_dir = 1
            print(INFO + f"Y direction: Y-UP  (head={head_y:.2f} > heel={heel_y:.2f})")
        else:
            self.up_dir = -1
            print(INFO + f"Y direction: Y-DOWN (head={head_y:.2f} < heel={heel_y:.2f})")

        self.df.interpolate(method="linear", limit_direction="both", inplace=True)
        self.filter_data()

        # Fix 1: compute actual walking and lateral axes from heel trajectory
        self._compute_walk_axes()

        self._load_raw2d(csv_path)

    # ─── Raw 2D companion data (for MLE-refined planted points) ──────────────

    def _load_raw2d(self, csv_path):
        """
        Loads the "<csv>_raw2d.npz" companion written by pose_estimation_v2.py
        (per-camera 2D foot keypoints/scores, camera projection matrices, and
        the floor plane used for alignment). If absent (e.g. CSV produced by
        v1 pose estimation), MLE refinement is skipped and _get_planted_pt
        falls back to the local height-minimum heuristic.
        """
        raw2d_path = os.path.splitext(csv_path)[0] + "_raw2d.npz"
        self.raw2d = None
        if not os.path.exists(raw2d_path):
            print(INFO + "No raw2D companion file — using height-minimum heuristic for planted points.")
            return

        npz = np.load(raw2d_path)
        self.raw2d = npz["raw2d"]                 # (n_frames, n_cams, n_feet_joints, 3)
        self.feet_joint_ids = npz["feet_indices"].tolist()
        self.cam_P = npz["cam_P"]                 # (n_cams, 3, 4)
        self.R_fix = npz["R_fix"]
        self.floor_offset = float(npz["floor_offset"])
        self.plane_n, self.plane_d = ground_plane_from_alignment(self.R_fix, self.floor_offset)
        print(INFO + f"Loaded raw2D companion — MLE-refined planted points enabled: {raw2d_path}")

    # ─── Fix 1: Walking-direction PCA ────────────────────────────────────────

    def _compute_walk_axes(self):
        """
        PCA on combined L+R heel positions in the XZ plane.
        Sets self.walk_dir and self.lat_dir — unit 2-vectors (x, z).
        Falls back to (Z-forward, X-lateral) if not enough valid data.
        """
        lx = self.df["j19_x"].values
        lz = self.df["j19_z"].values
        rx = self.df["j22_x"].values
        rz = self.df["j22_z"].values

        xz = np.column_stack([np.concatenate([lx, rx]), np.concatenate([lz, rz])])
        valid = ~np.isnan(xz).any(axis=1)
        xz = xz[valid]

        if len(xz) < 4:
            self.walk_dir = np.array([0.0, 1.0])
            self.lat_dir = np.array([1.0, 0.0])
            print(
                INFO + "Walk-axis PCA: too few heel points — defaulting to Z-forward."
            )
            return

        _, _, vh = np.linalg.svd(xz - xz.mean(axis=0), full_matrices=False)
        self.walk_dir = vh[0]  # highest-variance component = walking direction
        self.lat_dir = vh[1]  # lowest-variance component  = lateral

        angle = float(np.degrees(np.arctan2(self.walk_dir[0], self.walk_dir[1])))
        print(
            INFO + f"Walk-axis PCA: walking direction {angle:.1f}° from Z-axis  "
            f"(walk={self.walk_dir.round(3).tolist()}  "
            f"lat={self.lat_dir.round(3).tolist()})"
        )

    # ─── Skeleton filter (unchanged from v1) ─────────────────────────────────

    def filter_data(self):
        b, a = butter(4, 6 / (0.5 * self.fps), btype="low")
        for col in self.df.columns:
            if col.startswith("j"):
                self.df[col] = self.df[col].ffill().bfill()
                self.df[col] = filtfilt(b, a, self.df[col])

    # ─── Fix 3: Tighter event-detection filter ───────────────────────────────

    def _event_filter(self, arr):
        """
        Apply a second 4 Hz low-pass to a heel/toe height array.
        The skeleton is already filtered at 6 Hz; this tighter pass removes
        residual keypoint jitter before find_peaks without blurring gait events.
        """
        cutoff = min(4.0, 0.45 * self.fps)
        b, a = butter(4, cutoff / (0.5 * self.fps), btype="low")
        return filtfilt(b, a, arr)

    # ─── Fix 2: Sub-frame peak refinement ────────────────────────────────────

    def _refine_peaks(self, signal, peaks):
        """
        Parabolic interpolation of detected peak positions to sub-frame precision.
        Returns a float array of fractional frame indices.

        For a peak at integer index k, fits a parabola through signal[k-1:k+2]
        and solves for the vertex.  The sub-frame offset is clamped to ±0.5 so
        the refined position stays within the original frame interval.
        """
        refined = np.empty(len(peaks), dtype=float)
        for i, idx in enumerate(peaks):
            if 1 <= idx < len(signal) - 1:
                y0, y1, y2 = signal[idx - 1], signal[idx], signal[idx + 1]
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-12:
                    delta = 0.5 * (y0 - y2) / denom
                    refined[i] = idx + float(np.clip(delta, -0.5, 0.5))
                else:
                    refined[i] = float(idx)
            else:
                refined[i] = float(idx)
        return refined

    # ─── Floor-collision crossing detector ───────────────────────────────────

    def _floor_crossings(self, y_signal, floor_level, threshold=0.05,
                         min_contact_s=0.15, boundary_enter=False):
        """
        Find floor contact intervals from a height signal using threshold crossings.

        For Y-up  (up_dir=+1): "on floor" = y < floor_level + threshold
        For Y-down (up_dir=-1): "on floor" = y > floor_level - threshold

        boundary_enter : if True and the signal starts below threshold, insert a
            synthetic entry at frame 0 so the first exit (toe-off) is not lost.
            Should be False for heel strikes — those must come from actual crossings.

        Returns (strike_frames, off_frames) — float arrays of fractional frame
        indices sub-linearly interpolated at each exact crossing.  Only contact
        intervals longer than min_contact_s are kept (noise filter).
        """
        n = len(y_signal)
        if self.up_dir == 1:
            contact_thr = floor_level + threshold
            on_floor = y_signal < contact_thr
        else:
            contact_thr = floor_level - threshold
            on_floor = y_signal > contact_thr

        min_frames = max(2, int(min_contact_s * self.fps))

        # enter: False→True (foot coming down); leave: True→False (foot going up)
        enter_idx = np.where(~on_floor[:-1] &  on_floor[1:])[0]
        leave_idx = np.where( on_floor[:-1] & ~on_floor[1:])[0]

        # Boundary: only for toe-off detection (boundary_enter=True).
        if boundary_enter and on_floor[0]:
            enter_idx = np.concatenate([[0], enter_idx])

        strikes, offs = [], []

        for ef in enter_idx:
            # For synthetic frame-0 entry: include leave at frame 0 (1-frame contact
            # → rejected by duration filter, prevents latching onto the next real leave)
            min_lf = ef if ef == 0 else ef + 1
            lf_candidates = leave_idx[leave_idx >= min_lf]
            lf = lf_candidates[0] if len(lf_candidates) > 0 else n - 2

            if (lf - ef) < min_frames:
                continue

            # Sub-frame entering crossing (ef → ef+1); at frame 0 boundary no interpolation
            if ef == 0:
                strikes.append(0.0)
            else:
                y0, y1 = y_signal[ef], y_signal[ef + 1]
                dy = y1 - y0
                t = (contact_thr - y0) / dy if abs(dy) > 1e-9 else 0.0
                strikes.append(ef + float(np.clip(t, 0.0, 1.0)))

            # Sub-frame leaving crossing between frame lf and lf+1
            lf1 = min(lf + 1, n - 1)
            y0, y1 = y_signal[lf], y_signal[lf1]
            dy = y1 - y0
            t = (contact_thr - y0) / dy if abs(dy) > 1e-9 else 0.0
            offs.append(lf + float(np.clip(t, 0.0, 1.0)))

        return np.array(strikes, dtype=float), np.array(offs, dtype=float)

    # ─── Heel-strike timing refinement ───────────────────────────────────────

    def _refine_strikes(self, y_signal, floor_level, broad_strikes, refine_thr=0.02):
        """
        Two-pass heel strike refinement.

        The broad threshold (5 cm) reliably identifies genuine contacts.
        Within each detected contact, search forward for the first crossing
        of the finer threshold (2 cm) — much closer to actual ground contact.
        Falls back to the broad crossing if no fine crossing is found within
        half a stride (≈12 frames at 25 fps).
        """
        fine_thr = floor_level + refine_thr if self.up_dir == 1 else floor_level - refine_thr
        n = len(y_signal)
        refined = []
        for s in broad_strikes:
            lo = max(0, int(np.floor(s)))
            found = False
            for i in range(lo, min(lo + 12, n - 1)):
                y0, y1 = y_signal[i], y_signal[i + 1]
                crosses = (y0 >= fine_thr and y1 < fine_thr) if self.up_dir == 1 \
                     else (y0 <= fine_thr and y1 > fine_thr)
                if crosses:
                    dy = y1 - y0
                    t = (fine_thr - y0) / dy if abs(dy) > 1e-9 else 0.0
                    refined.append(i + float(np.clip(t, 0.0, 1.0)))
                    found = True
                    break
            if not found:
                refined.append(float(s))   # fall back to broad crossing
        return np.array(refined, dtype=float)

    # ─── Planted foot position (spatial measurement decoupled from timing) ──────

    def _get_planted_pt(self, strike_frame, side):
        """
        Return the heel's ground-contact XYZ within 8 frames after the
        detected strike.

        When a raw2D companion is loaded, pools every camera's 2D heel
        observation across that window into one ground-plane-constrained MLE
        solve (foot_point_mle.estimate_stance_point_mle) — the foot is assumed
        static while planted, so every (camera, frame) detection is evidence
        for the same point. Otherwise falls back to the local height-minimum
        heuristic on the aligned 3D CSV: the strike frame sets WHEN contact
        began; the minimum sets WHERE the foot actually landed (after the
        foot finishes settling).
        """
        lo = max(0, int(np.floor(strike_frame)))
        hi = min(lo + 8, len(self.df) - 1)

        if self.raw2d is not None:
            mle_pt = self._mle_planted_pt(lo, hi, side)
            if mle_pt is not None:
                return mle_pt

        col_y = self.map[f"{side}_Heel_{self.height_axis.upper()}"]
        window = self.df[col_y].values[lo : hi + 1]
        min_off = int(np.argmin(window)) if self.up_dir == 1 else int(np.argmax(window))
        return self._get_pt_at(float(lo + min_off), f"{side}_Heel")

    def _mle_planted_pt(self, lo, hi, side):
        """
        Pools raw 2D heel observations (all cameras, frames lo..hi) into a
        ground-plane MLE solve, and converts the result from the raw
        triangulation frame back into the aligned/flipped coordinate frame
        used everywhere else in this class. Returns None if too few valid
        2D observations are available in the window (caller falls back).
        """
        joint_id = 19 if side == "L" else 22   # *_Heel
        slot = self.feet_joint_ids.index(joint_id)
        frame_ids = self.df["frame_idx"].values

        frame_obs = []
        for f in range(lo, hi + 1):
            if f >= len(frame_ids):
                continue
            orig_f = int(frame_ids[f])
            if orig_f >= len(self.raw2d):
                continue
            obs = []
            for cam_idx in range(self.raw2d.shape[1]):
                u, v, score = self.raw2d[orig_f, cam_idx, slot]
                if not np.isnan(u):
                    obs.append((self.cam_P[cam_idx], (u, v), score))
            if obs:
                frame_obs.append(obs)

        if sum(len(o) for o in frame_obs) < 2:
            return None

        result = estimate_stance_point_mle(frame_obs, self.plane_n, self.plane_d)
        X_aligned = result["X"] @ self.R_fix.T
        X_aligned[1] -= self.floor_offset
        X_aligned[0] = -X_aligned[0]
        return X_aligned

    # ─── Interpolated keypoint lookup (for fractional frame positions) ────────

    def _get_pt_at(self, frac_frame, name):
        """
        Linearly interpolated XYZ position of a keypoint at a fractional frame index.
        Used so that stride-start/end positions match sub-frame event times.
        """
        lo = int(np.floor(frac_frame))
        hi = min(lo + 1, len(self.df) - 1)
        t = frac_frame - lo

        def _row(r):
            return np.array(
                [
                    self.df.iloc[r][self.map[f"{name}_X"]],
                    self.df.iloc[r][self.map[f"{name}_Y"]],
                    self.df.iloc[r][self.map[f"{name}_Z"]],
                ]
            )

        return _row(lo) * (1.0 - t) + _row(hi) * t

    # ─── Event detection — floor-collision crossing ───────────────────────────

    def detect_events(self, side):
        """
        Returns fractional frame indices (float arrays) for heel strikes and toe-offs.

        Heel strike = heel Y crosses below floor+5 cm (entering ground contact).
        Toe-off     = toe  Y crosses above floor+5 cm (leaving ground contact).

        Both times are sub-frame interpolated at the exact threshold crossing.
        The 4 Hz event filter is applied first to suppress keypoint jitter.
        """
        col_heel = self.map[f"{side}_Heel_{self.height_axis.upper()}"]
        col_toe  = self.map[f"{side}_Toe_{self.height_axis.upper()}"]

        heel_y = self._event_filter(self.df[col_heel].ffill().bfill().values)
        toe_y  = self._event_filter(self.df[col_toe].ffill().bfill().values)

        heel_floor = float(np.nanpercentile(heel_y, 5))
        toe_floor  = float(np.nanpercentile(toe_y,  5))

        # Heel: broad 5 cm detects robust contact windows; refine to 2 cm for
        # accurate strike timing (5 cm fires ~3 frames early as foot descends).
        # Toe: 3 cm catches all contact periods reliably, including the first
        # contact where the filtered toe signal barely dips below 2 cm.
        HEEL_THR    = 0.05   # 5 cm broad — robust contact window detection
        HEEL_REFINE = 0.02   # 2 cm fine  — precise strike timing
        TOE_THR     = 0.03   # 3 cm       — toe contact detection and off timing
        MIN_CONTACT = 0.15   # discard contacts shorter than 150 ms (noise)

        broad_strikes, _ = self._floor_crossings(heel_y, heel_floor, HEEL_THR,  MIN_CONTACT,
                                                  boundary_enter=False)
        strikes = self._refine_strikes(heel_y, heel_floor, broad_strikes, HEEL_REFINE)
        _,       offs    = self._floor_crossings(toe_y,  toe_floor,  TOE_THR, MIN_CONTACT,
                                                  boundary_enter=True)

        print(INFO + f"  [{side}] floor-crossings: {len(strikes)} strikes  "
                     f"{len(offs)} toe-offs  "
                     f"(heel_floor={heel_floor:.3f} m  toe_floor={toe_floor:.3f} m)")

        return np.sort(strikes), np.sort(offs)

    # ─── Stride collection (all three fixes applied) ─────────────────────────

    def collect_strides(
        self,
        strikes,
        offs,
        opp_strikes,
        opp_offs,
        side,
        min_stride_dur=None,
        boundary_buffer=4,
    ):
        """
        Parameters
        ----------
        strikes / offs / opp_strikes / opp_offs : float arrays of fractional frame indices
        min_stride_dur : minimum stride duration in seconds (short strides are skipped)
        boundary_buffer : search window extension before stride start, in frames
        """
        if len(strikes) < 2:
            return []

        per_stride = []

        for i in range(len(strikes) - 1):
            start = strikes[i]
            end = strikes[i + 1]
            stride_dur = (end - start) / self.fps

            if stride_dur == 0:
                continue

            if min_stride_dur is not None and stride_dur < min_stride_dur:
                print(
                    INFO + f"  [{side}] stride {i}: {stride_dur:.2f}s < min "
                    f"{min_stride_dur:.2f}s  → SKIPPED"
                )
                continue

            # Planted positions for longitudinal measurements: heel at its local
            # height minimum after each strike.  The minimum is when the foot is
            # fully settled — correct Z (walking direction) even when the broad
            # strike fires a few frames early during late swing.
            p1_plant = self._get_planted_pt(start, side)
            p2_plant = self._get_planted_pt(end,   side)

            # Stride length: Euclidean XZ displacement between consecutive planted contacts
            stride_len = float(np.sqrt((p2_plant[0] - p1_plant[0]) ** 2 + (p2_plant[2] - p1_plant[2]) ** 2))

            opp_side = "R" if side == "L" else "L"
            prior_opp = opp_strikes[opp_strikes < start]
            if len(prior_opp) > 0:
                opp_contact_f = prior_opp[-1]
                opp_plant = self._get_planted_pt(opp_contact_f, opp_side)

                # Step length: along walking direction — planted positions give correct Z
                diff_plant = np.array([p1_plant[0] - opp_plant[0], p1_plant[2] - opp_plant[2]])
                step_len = abs(float(np.dot(diff_plant, self.walk_dir)))

                # Step width: along lateral direction — use at-strike positions.
                # At initial contact the foot is in its most outward (lateral) position;
                # the settled mid-stance position is more medial and underestimates width.
                p1_strike   = self._get_pt_at(start,       f"{side}_Heel")
                opp_at_stk  = self._get_pt_at(opp_contact_f, f"{opp_side}_Heel")
                diff_strike = np.array([p1_strike[0] - opp_at_stk[0], p1_strike[2] - opp_at_stk[2]])
                step_width  = abs(float(np.dot(diff_strike, self.lat_dir)))
            else:
                step_len   = np.nan
                step_width = np.nan

            # Phase events — all comparisons in fractional frame units
            valid_offs = offs[(offs > start) & (offs < end)]
            foot_off_pct = np.nan
            if len(valid_offs) > 0:
                foot_off_pct = (valid_offs[0] - start) / (end - start) * 100.0

            valid_opp_s = opp_strikes[(opp_strikes > start) & (opp_strikes < end)]
            opp_con_pct = np.nan
            step_time = np.nan
            if len(valid_opp_s) > 0:
                opp_con_pct = (valid_opp_s[0] - start) / (end - start) * 100.0
                step_time = (valid_opp_s[0] - start) / self.fps

            # Extend search window backward by boundary_buffer frames for opp toe-off
            valid_opp_o = opp_offs[
                (opp_offs >= start - boundary_buffer) & (opp_offs < end)
            ]
            opp_off_pct = np.nan
            if len(valid_opp_o) > 0:
                raw_pct = (valid_opp_o[0] - start) / (end - start) * 100.0
                opp_off_pct = max(raw_pct, 0.0)

            single_supp = np.nan
            if not (np.isnan(opp_con_pct) or np.isnan(opp_off_pct)):
                single_supp = (opp_con_pct - opp_off_pct) / 100.0 * stride_dur

            double_supp = np.nan
            if not (
                np.isnan(foot_off_pct) or np.isnan(opp_con_pct) or np.isnan(opp_off_pct)
            ):
                double_supp = (
                    (opp_off_pct + (foot_off_pct - opp_con_pct)) / 100.0 * stride_dur
                )

            limp = np.nan
            if not np.isnan(foot_off_pct):
                swing = 100.0 - foot_off_pct
                if swing > 0:
                    limp = foot_off_pct / swing

            per_stride.append(
                {
                    "StrideTime": stride_dur,
                    "StrideLen": stride_len,
                    "StepLen": step_len,
                    "StepWidth": step_width,
                    "WalkingSpeed": stride_len / stride_dur,
                    "Cadence": (60.0 / stride_dur) * 2,
                    "StepTime": step_time,
                    "FootOff": foot_off_pct,
                    "OppFootContact": opp_con_pct,
                    "OppFootOff": opp_off_pct,
                    "SingleSupport": single_supp,
                    "DoubleSupport": double_supp,
                    "LimpIndex": limp,
                }
            )

        return per_stride

    # ─── Public interface (unchanged from v1) ────────────────────────────────

    def generate_vicon_tables(self, extra_csv_paths=None):
        """
        Returns (params_df, empty_df) compatible with v1 output.
        extra_csv_paths : list of additional CSV paths to pool strides from.
        """
        BOUNDARY_BUFFER = 4

        l_strikes, l_offs = self.detect_events("L")
        r_strikes, r_offs = self.detect_events("R")

        l_durs = [
            (l_strikes[i + 1] - l_strikes[i]) / self.fps
            for i in range(len(l_strikes) - 1)
        ]
        expected_stride = float(np.median(l_durs)) if l_durs else None
        min_stride_dur = expected_stride * 0.80 if expected_stride else None

        if expected_stride:
            print(
                INFO + f"Expected stride: {expected_stride:.3f}s  "
                f"→ min allowed: {min_stride_dur:.3f}s"
            )

        l_strides = self.collect_strides(
            l_strikes,
            l_offs,
            r_strikes,
            r_offs,
            "L",
            min_stride_dur=min_stride_dur,
            boundary_buffer=BOUNDARY_BUFFER,
        )
        r_strides = self.collect_strides(
            r_strikes,
            r_offs,
            l_strikes,
            l_offs,
            "R",
            min_stride_dur=min_stride_dur,
            boundary_buffer=BOUNDARY_BUFFER,
        )

        print(
            INFO
            + f"[primary]  L: {len(l_strides)} strides  R: {len(r_strides)} strides"
        )

        if extra_csv_paths:
            for extra_path in extra_csv_paths:
                print(INFO + f"Pooling: {extra_path}")
                try:
                    extra = GaitAnalyzer(extra_path, fps=self.fps)
                    el, elo = extra.detect_events("L")
                    er, ero = extra.detect_events("R")

                    e_durs = [
                        (el[i + 1] - el[i]) / self.fps for i in range(len(el) - 1)
                    ]
                    e_exp = float(np.median(e_durs)) if e_durs else expected_stride
                    e_min = e_exp * 0.80 if e_exp else min_stride_dur

                    new_l = extra.collect_strides(
                        el,
                        elo,
                        er,
                        ero,
                        "L",
                        min_stride_dur=e_min,
                        boundary_buffer=BOUNDARY_BUFFER,
                    )
                    new_r = extra.collect_strides(
                        er,
                        ero,
                        el,
                        elo,
                        "R",
                        min_stride_dur=e_min,
                        boundary_buffer=BOUNDARY_BUFFER,
                    )
                    l_strides += new_l
                    r_strides += new_r
                    print(INFO + f"  added L:{len(new_l)} R:{len(new_r)} strides")

                except Exception as exc:
                    print(INFO + f"  Warning — could not load {extra_path}: {exc}")

            print(
                INFO
                + f"[pooled]   L: {len(l_strides)} strides  R: {len(r_strides)} strides"
            )

        def to_means(strides):
            if not strides:
                return {}
            return {k: np.nanmean([s[k] for s in strides]) for k in strides[0]}

        l_res = to_means(l_strides)
        r_res = to_means(r_strides)

        param_defs = [
            ("Cadence", "Cadence", "steps/min"),
            ("WalkingSpeed", "Walking Speed", "m/s"),
            ("StrideTime", "Stride Time", "s"),
            ("StepTime", "Step Time", "s"),
            ("OppFootOff", "Opposite Foot Off", "%"),
            ("OppFootContact", "Opposite Foot Contact", "%"),
            ("FootOff", "Foot Off", "%"),
            ("SingleSupport", "Single Support", "s"),
            ("DoubleSupport", "Double Support", "s"),
            ("StrideLen", "Stride Length", "m"),
            ("StepLen", "Step Length", "m"),
            ("StepWidth", "Step Width", "m"),
            ("LimpIndex", "Limp Index", "ratio"),
        ]

        rows = []

        def add_rows(res, ctx):
            if not res:
                return
            for k, name, unit in param_defs:
                rows.append(
                    {
                        "Subject": SUBJECT_NAME,
                        "Context": ctx,
                        "Name": name,
                        "Value": res.get(k, np.nan),
                        "Units": unit,
                    }
                )

        add_rows(l_res, "Left")
        add_rows(r_res, "Right")

        return pd.DataFrame(rows), pd.DataFrame()


def main():
    START_FRAME = None   # None = use all frames from the beginning
    END_FRAME   = None   # None = use all frames to the end

    EXTRA_TRIALS = []

    analyzer = GaitAnalyzer(
        OUTPUT_CSV,
        fps=FPS_ANALYSIS,
        height_axis="y",
        start_frame=START_FRAME,
        end_frame=END_FRAME,
    )

    params_df, _ = analyzer.generate_vicon_tables(
        extra_csv_paths=EXTRA_TRIALS if EXTRA_TRIALS else None
    )

    print("\n# Gait Cycle Parameters")
    print(params_df.to_markdown(index=True))

    gait_out = "output_mahidol/gait_v2"
    os.makedirs(gait_out, exist_ok=True)
    save_path = os.path.join(
        gait_out, f"{DAY}-{MONTH}_{SUBJECT_NAME}_p{P_NO}_r{ROUND}_gait.csv"
    )
    params_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
