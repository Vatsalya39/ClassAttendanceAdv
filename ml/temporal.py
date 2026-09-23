"""
ml/temporal.py
──────────────
Module 4 / Step 4: Temporal tracking and embedding fusion across multi-frame bursts.

Classroom session captures can include a 3-5 frame burst captured within 1-2 seconds.
Students are seated in fixed desks; natural small movements (nodding, shifting,
blinking, brief motion blur) occur, but bounding-box continuity holds strongly across
burst frames (IoU >= 0.50).

This module:
1. Performs IoU-based multi-object tracking across sequential burst frames.
2. Fuses embeddings along confirmed tracklets via L2-normalised quality-weighted averaging.
3. Recovers clean identity representations when a momentary blink, blur, or turned head
   affects one frame of the burst while other frames remain sharp and frontal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ml.pipeline import BoundingBox, FaceEmbedding


def compute_bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = float(inter_w * inter_h)
    if inter <= 0.0:
        return 0.0
    union = float(a.area + b.area - inter)
    return inter / union if union > 0.0 else 0.0


@dataclass
class BurstTracklet:
    """A series of detections associated with a single individual across burst frames."""

    track_id: int
    # List of (frame_index, FaceEmbedding)
    observations: list[tuple[int, FaceEmbedding]] = field(default_factory=list)

    @property
    def latest_bbox(self) -> BoundingBox:
        assert self.observations, "Cannot get bbox of empty tracklet"
        return self.observations[-1][1].bbox

    @property
    def representative_bbox(self) -> BoundingBox:
        """Choose the bbox from the highest detection score observation."""
        best_obs = max(self.observations, key=lambda obs: obs[1].detection_score)
        return best_obs[1].bbox

    @property
    def frame_count(self) -> int:
        return len(self.observations)


def track_burst_faces(
    burst_frames: Sequence[Sequence[FaceEmbedding]],
    iou_threshold: float = 0.50,
) -> list[BurstTracklet]:
    """
    Associate face detections across sequential burst frames using IoU tracking.

    Parameters
    ----------
    burst_frames : Sequence of detection lists, one per burst frame (0 to N-1).
    iou_threshold : Minimum IoU to associate detections between frame t and t+1.
                    Default 0.50 reliably separates the same seated student
                    under small posture shifts from adjacent desk neighbors.

    Returns
    -------
    List of BurstTracklet objects containing associated detections.
    """
    if not burst_frames:
        return []

    tracklets: list[BurstTracklet] = []
    active_tracklets: list[BurstTracklet] = []
    next_track_id = 1

    # Initialize with frame 0
    for face in burst_frames[0]:
        t = BurstTracklet(track_id=next_track_id, observations=[(0, face)])
        next_track_id += 1
        tracklets.append(t)
        active_tracklets.append(t)

    # Sequentially track through frames 1 to N-1
    for frame_idx in range(1, len(burst_frames)):
        current_faces = list(burst_frames[frame_idx])
        if not current_faces:
            continue

        if not active_tracklets:
            # No prior active tracks; start new tracks for all faces in this frame
            for face in current_faces:
                t = BurstTracklet(track_id=next_track_id, observations=[(frame_idx, face)])
                next_track_id += 1
                tracklets.append(t)
                active_tracklets.append(t)
            continue

        # Compute IoU cost matrix between active tracklets and current detections
        n_tracks = len(active_tracklets)
        n_faces = len(current_faces)
        iou_matrix = np.zeros((n_tracks, n_faces), dtype=np.float32)

        for i, track in enumerate(active_tracklets):
            last_box = track.latest_bbox
            for j, face in enumerate(current_faces):
                iou_matrix[i, j] = compute_bbox_iou(last_box, face.bbox)

        # Greedy bipartite matching on highest IoU
        matched_tracks: set[int] = set()
        matched_faces: set[int] = set()

        # Sort candidate matches by descending IoU
        candidate_pairs = []
        for i in range(n_tracks):
            for j in range(n_faces):
                if iou_matrix[i, j] >= iou_threshold:
                    candidate_pairs.append((float(iou_matrix[i, j]), i, j))
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        for _, track_idx, face_idx in candidate_pairs:
            if track_idx not in matched_tracks and face_idx not in matched_faces:
                active_tracklets[track_idx].observations.append(
                    (frame_idx, current_faces[face_idx])
                )
                matched_tracks.add(track_idx)
                matched_faces.add(face_idx)

        # Unmatched detections become new tracklets
        new_active: list[BurstTracklet] = [active_tracklets[i] for i in matched_tracks]
        for j, face in enumerate(current_faces):
            if j not in matched_faces:
                t = BurstTracklet(track_id=next_track_id, observations=[(frame_idx, face)])
                next_track_id += 1
                tracklets.append(t)
                new_active.append(t)

        active_tracklets = new_active

    return tracklets


def fuse_tracklet_embeddings(tracklet: BurstTracklet) -> FaceEmbedding:
    """
    Fuse the embeddings of a tracklet across burst frames.

    Features:
    1. Weighted vector average: weights proportional to detection confidence.
    2. Quality recovery: if at least one burst frame passed quality gates without
       blur or pose flags, the clean frames are prioritized and flags cleared.
    3. L2-normalisation: strictly preserves unit-norm property required by cosine distance.
    """
    assert tracklet.observations, "Cannot fuse empty tracklet"
    if len(tracklet.observations) == 1:
        return tracklet.observations[0][1]

    # Check if there are clean observations (no quality flags)
    clean_obs = [obs for obs in tracklet.observations if len(obs[1].quality_flags) == 0]
    candidate_obs = clean_obs if clean_obs else tracklet.observations

    embeddings = [np.array(obs[1].embedding, dtype=np.float32) for obs in candidate_obs]
    weights = [max(0.1, obs[1].detection_score) for obs in candidate_obs]

    # Weighted sum
    fused_vec = np.zeros(512, dtype=np.float32)
    for emb, w in zip(embeddings, weights, strict=False):
        fused_vec += emb * w

    # L2-normalise
    norm = float(np.linalg.norm(fused_vec))
    if norm > 1e-12:
        fused_vec /= norm
    else:
        fused_vec = embeddings[0]

    # Flags: if clean observation existed, flags are empty; otherwise take intersection
    final_flags: tuple[str, ...]
    if clean_obs:
        final_flags = ()
    else:
        # Common quality failures across all frames
        common = set(tracklet.observations[0][1].quality_flags)
        for obs in tracklet.observations[1:]:
            common.intersection_update(obs[1].quality_flags)
        final_flags = tuple(sorted(common))

    return FaceEmbedding(
        bbox=tracklet.representative_bbox,
        embedding=[float(v) for v in fused_vec],
        quality_flags=final_flags,
        detection_score=max(obs[1].detection_score for obs in tracklet.observations),
    )


def fuse_burst_detections(
    burst_frames: Sequence[Sequence[FaceEmbedding]],
    iou_threshold: float = 0.50,
) -> list[FaceEmbedding]:
    """
    High-level entry point: Track faces across a burst and return fused FaceEmbedding objects.
    """
    tracklets = track_burst_faces(burst_frames, iou_threshold=iou_threshold)
    return [fuse_tracklet_embeddings(t) for t in tracklets]
