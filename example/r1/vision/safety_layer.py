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
    allow_turn_in_place: bool = False
    robot_radius_m: float = 0.28
    safety_margin_m: float = 0.10
    trajectory_horizon_s: float = 1.5
    trajectory_step_s: float = 0.10


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

        if requested.is_turn_in_place and self.config.allow_turn_in_place:
            return SafetyDecision(requested, "in-place turn allowed")

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

        self._accepted_count += 1
        confirmed = self._accepted_count >= self.config.clear_confirmations
        if not confirmed:
            return self._stop("waiting for perception confirmation")

        return self._select_command(requested, snapshot.obstacle_points)

    def _select_command(self, requested: VelocityCommand, obstacle_points) -> SafetyDecision:
        points = self._points(obstacle_points)
        candidates = self._candidates(requested)
        safe = [command for command in candidates if self._is_safe(command, points)]
        if not safe:
            return self._stop("no collision-free local trajectory")

        command = min(safe, key=lambda item: self._cost(item, requested))
        yaw_change = command.yaw_rate - requested.yaw_rate
        if abs(yaw_change) < 1e-6:
            self._avoidance_sign = 0
            return SafetyDecision(command, "requested trajectory clear")

        self._avoidance_sign = 1 if yaw_change > 0.0 else -1
        direction = "left" if self._avoidance_sign > 0 else "right"
        return SafetyDecision(command, f"avoiding obstacle: {direction}")

    def _points(self, obstacle_points):
        if obstacle_points is None:
            return np.empty((0, 2), dtype=np.float32)
        points = np.asarray(obstacle_points, dtype=np.float32)
        return points.reshape((-1, 2))

    def _candidates(self, requested: VelocityCommand):
        speed = requested.vx * 0.75
        candidates = [requested]
        for yaw_change in (-1.2, -0.7, 0.7, 1.2):
            direction = "LEFT" if yaw_change > 0.0 else "RIGHT"
            candidates.append(VelocityCommand(speed, 0.0, requested.yaw_rate + yaw_change, f"AVOID {direction}"))
        return candidates

    def _is_safe(self, command: VelocityCommand, points) -> bool:
        if len(points) == 0:
            return True

        radius = self.config.robot_radius_m + self.config.safety_margin_m
        position = np.zeros(2, dtype=np.float32)
        heading = 0.0
        steps = int(np.ceil(self.config.trajectory_horizon_s / self.config.trajectory_step_s))
        for _ in range(steps + 1):
            if np.any(np.sum((points - position) ** 2, axis=1) <= radius ** 2):
                return False
            heading += command.yaw_rate * self.config.trajectory_step_s
            position += command.vx * self.config.trajectory_step_s * np.array(
                (np.cos(heading), np.sin(heading)), dtype=np.float32
            )
        return True

    def _cost(self, command: VelocityCommand, requested: VelocityCommand) -> float:
        yaw_change = command.yaw_rate - requested.yaw_rate
        side_change = 0 if yaw_change == 0.0 else (1 if yaw_change > 0.0 else -1)
        side_penalty = 0.05 if self._avoidance_sign and side_change != self._avoidance_sign else 0.0
        return abs(command.vx - requested.vx) + 0.20 * abs(yaw_change) + side_penalty

    def _stop(self, reason: str) -> SafetyDecision:
        return SafetyDecision(VelocityCommand.stop(), reason)
