"""Tracking wrapper that stabilizes card identities across frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from common.card import Card
from cv_pipeline.detection.inference import Detection
from cv_pipeline.detection.utils import iou


CONFIRMED_THRESHOLD = 10  # frames needed to confirm a card
NEW_CARD_THRESHOLD = 5    # consecutive mismatched frames before accepting a new card on top


@dataclass
class TrackedCard:
    """Tracked card state tied to a persistent track ID."""

    track_id: int
    card: Card
    bbox: Tuple[int, int, int, int]
    confidence: float
    missed_frames: int = 0
    seen_frames: int = 1
    confirmed: bool = False
    # Tracks a candidate new-card label appearing at this position
    candidate_label: str | None = None
    candidate_frames: int = 0


class ByteTrackWrapper:
    """Simple IoU-based tracker API compatible with ByteTrack-like usage."""

    def __init__(self, max_missed_frames: int = 10, iou_match_threshold: float = 0.3) -> None:
        """Initialize tracker state.

        Args:
            max_missed_frames: Frames to keep unmatched tracks before dropping.
            iou_match_threshold: Minimum IoU to match detection to an existing track.
        """
        self.max_missed_frames = max_missed_frames
        self.iou_match_threshold = iou_match_threshold
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedCard] = {}

    def update(self, detections: List[Detection]) -> List[TrackedCard]:
        """Update tracks from detector outputs.

        Args:
            detections: Detection objects from current frame.

        Returns:
            Active tracks including temporarily occluded entries (< max_missed_frames).
        """
        unmatched_track_ids = set(self.tracks.keys())

        for det in detections:
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                score = iou(track.bbox, det.bbox)
                if score > best_iou and score >= self.iou_match_threshold:
                    best_iou = score
                    best_track_id = track_id

            det_card = Card.from_label(det.label)

            # If matched a confirmed track but label differs, a new card may
            # have been placed on top.  Only accept it after the new label is
            # stable for NEW_CARD_THRESHOLD consecutive frames to filter out
            # classifier flicker.
            if best_track_id is not None:
                matched_track = self.tracks[best_track_id]
                if matched_track.confirmed and det_card.label != matched_track.card.label:
                    if matched_track.candidate_label == det_card.label:
                        matched_track.candidate_frames += 1
                    else:
                        matched_track.candidate_label = det_card.label
                        matched_track.candidate_frames = 1

                    if matched_track.candidate_frames >= NEW_CARD_THRESHOLD:
                        # Stable new label — accept as a genuinely new card
                        matched_track.candidate_label = None
                        matched_track.candidate_frames = 0
                        best_track_id = None
                    else:
                        # Not yet stable — keep matching to existing track
                        unmatched_track_ids.discard(best_track_id)
                        continue

            if best_track_id is None:
                self.tracks[self.next_track_id] = TrackedCard(
                    track_id=self.next_track_id,
                    card=det_card,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    missed_frames=0,
                )
                self.next_track_id += 1
                continue

            track = self.tracks[best_track_id]
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.candidate_label = None
            track.candidate_frames = 0
            if not track.confirmed:
                track.card = det_card
                track.card.is_counted = self.tracks[best_track_id].card.is_counted
            track.missed_frames = 0
            track.seen_frames += 1
            if track.seen_frames >= CONFIRMED_THRESHOLD:
                track.confirmed = True
            unmatched_track_ids.discard(best_track_id)

        # If no detections at all, the table is clear — reset all tracks
        if not detections:
            self.tracks.clear()
            return []

        # Only drop unconfirmed tracks after max_missed_frames
        # Confirmed tracks persist until the table clears
        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed_frames += 1
            if not track.confirmed and track.missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        return list(self.tracks.values())
