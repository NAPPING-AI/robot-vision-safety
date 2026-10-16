from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PerceptionState(str, Enum):
    CLEAR = "NO OBSTACLE DETECTED"
    OBSTACLE = "OBSTACLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity requested by a policy."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    label: str = "STOP"
    desired_heading: float | None = None 

    @property
    def is_stop(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.yaw_rate == 0.0

    @property
    def is_turn_in_place(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.yaw_rate != 0.0

    @classmethod
    def stop(cls, label: str = "STOP") -> VelocityCommand:
        return cls(label=label)


@dataclass(frozen=True)
class PerceptionSnapshot:
    state: PerceptionState
    measurement_time: float
    obstacle_points: object = None


@dataclass(frozen=True)
class SafetyConfig:
    max_measurement_age: float
    clear_confirmations: int = 1
    allow_reverse: bool = False
    allow_turn_in_place: bool = True
    robot_radius_m: float = 0.28
    safety_margin_m: float = 0.10
    trajectory_horizon_s: float = 1.5
    trajectory_step_s: float = 0.10
    yaw_step: float = 0.10
    max_yaw_change: float = 1.20
    turn_weight: float = 0.20
    clearance_weight: float = 0.04
    preferred_clearance_m: float = 0.25
    side_change_penalty: float = 0.001


@dataclass(frozen=True)
class SafetyDecision:
    command: VelocityCommand
    reason: str


class FailClosedVelocityShield:
    def __init__(self, config: SafetyConfig):
        if config.max_measurement_age <= 0:
            raise ValueError("max_measurement_age must be positive")
        if config.clear_confirmations < 1:
            raise ValueError("clear_confirmations must be at least one")
        if min(config.yaw_step, config.max_yaw_change,
               config.preferred_clearance_m, config.trajectory_horizon_s,
               config.trajectory_step_s) <= 0:
            raise ValueError("sampling intervals and distances must be positive")
        if min(config.turn_weight, config.clearance_weight, config.side_change_penalty) < 0:
            raise ValueError("score weights cannot be negative")
        self.config = config
        self._accepted_count = 0
        self._avoidance_sign = 0

    def reset(self) -> None:
        self._accepted_count = 0
        self._avoidance_sign = 0

    def evaluate(self, requested: VelocityCommand, snapshot: PerceptionSnapshot | None, now: float) -> SafetyDecision:
        """Return a safe command"""
        if requested.is_stop:
            return SafetyDecision(requested, "policy requested stop")

        if requested.vx < 0.0 and not self.config.allow_reverse:
            return self._stop("reverse is disabled")

        if requested.vy != 0.0:
            return self._stop("lateral motion is disabled")

        if requested.is_turn_in_place and not self.config.allow_turn_in_place:
            return self._stop("in-place turns are disabled")

        if snapshot is None:
            self.reset()
            return self._stop("no accepted perception")

        if now - snapshot.measurement_time > self.config.max_measurement_age:
            self.reset()
            return self._stop("perception is stale")

        if snapshot.state is PerceptionState.UNKNOWN:
            self.reset()
            return self._stop("perception is unknown")

        if snapshot.state not in {PerceptionState.CLEAR, PerceptionState.OBSTACLE}:
            self.reset()
            return self._stop("unrecognised perception state")

        points = self._points(snapshot.obstacle_points)
        if points is None:
            return self._stop("missing or invalid obstacle points")
        if requested.is_turn_in_place:
            if self._clearance(requested, points) <= 0:
                return self._stop("obstacle inside protected radius")
            return SafetyDecision(requested, "in-place turn")

        self._accepted_count += 1
        confirmed = self._accepted_count >= self.config.clear_confirmations
        if not confirmed:
            return self._stop("waiting for perception confirmation")

        return self._select_command(requested, points)

    def _select_command(self, requested: VelocityCommand, obstacle_points) -> SafetyDecision:
        points = self._points(obstacle_points)
        candidates = self._candidates(requested)
        safe = [(command, self._clearance(command, points)) for command in candidates]
        safe = [(command, gap) for command, gap in safe if gap > 0]
        if not safe:
            return self._stop("no collision-free local trajectory")

        command, gap = min(safe, key=lambda item: self._cost(item[0], requested, item[1]))
        yaw_change = command.yaw_rate - requested.yaw_rate
        if abs(yaw_change) < 1e-6:
            self._avoidance_sign = 0
            return SafetyDecision(command, "requested trajectory clear")

        self._avoidance_sign = 1 if yaw_change > 0.0 else -1
        direction = "left" if self._avoidance_sign > 0 else "right"
        return SafetyDecision(command, f"avoid {direction}: delta={yaw_change:+.1f}rad/s; clearance={gap:.2f}m")

    def _points(self, obstacle_points):
        if obstacle_points is None:
            return None
        try:
            points = np.asarray(obstacle_points, dtype=np.float32)
        except (ValueError, TypeError):
            return None
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            return None
        return points

    def _candidates(self, requested: VelocityCommand):
        speed = requested.vx * 0.75
        candidates = [requested]
        amounts = np.arange(self.config.yaw_step, self.config.max_yaw_change, self.config.yaw_step)
        amounts = np.unique(np.r_[amounts, self.config.max_yaw_change])
        for yaw_change in np.r_[-amounts[::-1], amounts]:
            direction = "LEFT" if yaw_change > 0.0 else "RIGHT"
            candidates.append(VelocityCommand(speed, 0.0, requested.yaw_rate + float(yaw_change), f"AVOID {direction}"))
        return candidates

    def _is_safe(self, command: VelocityCommand, points) -> bool:
        return self._clearance(command, points) > 0

    def _clearance(self, command, points):
        """Minimum predicted distance outside the original inflated circle."""
        if len(points) == 0:
            return self.config.preferred_clearance_m

        radius = self.config.robot_radius_m + self.config.safety_margin_m
        position = np.zeros(2, dtype=np.float32)
        heading = 0.0
        minimum = float("inf")
        steps = int(np.ceil(self.config.trajectory_horizon_s / self.config.trajectory_step_s))
        for _ in range(steps + 1):
            distance = float(np.sqrt(np.min(np.sum((points - position) ** 2, axis=1))))
            minimum = min(minimum, distance - radius)
            heading += command.yaw_rate * self.config.trajectory_step_s
            position += command.vx * self.config.trajectory_step_s * np.array(
                (np.cos(heading), np.sin(heading)), dtype=np.float32
            )
        return minimum

    def _cost(self, command: VelocityCommand, requested: VelocityCommand, clearance: float) -> float:
        yaw_change = command.yaw_rate - requested.yaw_rate
        side_change = 0 if yaw_change == 0.0 else (1 if yaw_change > 0.0 else -1)
        switched = self._avoidance_sign and side_change and side_change != self._avoidance_sign
        side_penalty = self.config.side_change_penalty if switched else 0.0
        room = float(np.clip(clearance / self.config.preferred_clearance_m, 0, 1))
        return (abs(command.vx - requested.vx)
                + self.config.turn_weight * abs(yaw_change)
                + self.config.clearance_weight * (1 - room)
                + side_penalty)

    def _stop(self, reason: str) -> SafetyDecision:
        return SafetyDecision(VelocityCommand.stop(), reason)
