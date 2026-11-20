#!/usr/bin/env python
"""在 Dogfight 2 中实时显示 JSBSim Heading 策略的固定四阶段飞行。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import evaluate_lowlevel_controller as heading


STAGE_LABELS = {
    "level_hold": "平飞保持",
    "turn_and_climb": "转弯爬升",
    "reverse_turn_and_slow": "反向转弯并减速",
    "recover_level": "恢复平飞",
}
SOCKET_TIMEOUT_SEC = 3.0


class DogfightClient:
    """按发行包 socket_lib 的四字节大端长度加 JSON 消息格式通信。"""

    def __init__(self, connection: socket.socket):
        self.connection = connection

    def _receive_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = self.connection.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Dogfight 已断开连接")
            data.extend(chunk)
        return bytes(data)

    def _send_json(self, message) -> None:
        payload = json.dumps(message, allow_nan=False).encode("utf-8")
        self.connection.sendall(len(payload).to_bytes(4, "big") + payload)

    def command(self, command: str, *, reply: bool = False, **args):
        try:
            self._send_json({"command": command, "args": args})
            if reply:
                size = int.from_bytes(self._receive_exact(4), "big")
                if not 0 < size <= 1024 * 1024:
                    raise ConnectionError(f"Dogfight 回复的消息长度无效（{size} 字节）")
                return json.loads(self._receive_exact(size))
        except TimeoutError as error:
            raise TimeoutError(f"Dogfight 处理 {command} 超时（{SOCKET_TIMEOUT_SEC:g} 秒）") from error

    def disconnect(self) -> None:
        # 发行包将 JSON 空字符串识别为正常断开，随后会重新监听下一次连接。
        try:
            self._send_json("")
        except OSError:
            pass

    def planes(self) -> list[str]:
        return self.command("GET_PLANESLIST", reply=True)

    def plane_state(self, plane_id: str) -> dict:
        state = self.command("GET_PLANE_STATE", reply=True, plane_id=plane_id)
        if not state["active"] or state["wreck"] or state["crashed"]:
            raise RuntimeError(f"Dogfight 中的 {plane_id} 已停用或损毁，请重启 Network mode")
        return state

    def use_external_pose(self, plane_id: str) -> None:
        self.plane_state(plane_id)
        for command in ("DEACTIVATE_IA", "DEACTIVATE_AUTOPILOT", "DEACTIVATE_USER_CONTROL"):
            self.command(command, machine_id=plane_id)
        self.command("RETRACT_GEAR", plane_id=plane_id)
        self.command("SET_MACHINE_CUSTOM_PHYSICS_MODE", machine_id=plane_id, flag=True)
        state = self.command("GET_MACHINE_CUSTOM_PHYSICS_MODE", reply=True, machine_id=plane_id)
        if not state["custom_physics_mode"]:
            raise RuntimeError(f"Dogfight 未为 {plane_id} 启用外部位姿")

    def send_pose(self, plane_id: str, pose: Pose) -> None:
        self.command(
            "UPDATE_MACHINE_KINETICS", machine_id=plane_id,
            matrix=pose.matrix(), v_move=pose.velocity.tolist(),
        )

    def confirm_pose(self, plane_id: str, pose: Pose) -> dict:
        # 网络线程接收命令与渲染线程应用位姿之间可能隔一帧。
        deadline = time.perf_counter() + SOCKET_TIMEOUT_SEC
        while True:
            state = self.plane_state(plane_id)
            if (
                np.linalg.norm(np.asarray(state["position"]) - pose.position) < 1.0
                and np.linalg.norm(np.asarray(state["move_vector"]) - pose.velocity) < 1.0
            ):
                return state
            if time.perf_counter() >= deadline:
                raise RuntimeError(f"Dogfight 未应用发送给 {plane_id} 的位姿或速度")
            time.sleep(0.025)


def dogfight_quaternion(rpy: np.ndarray) -> np.ndarray:
    """将 JSBSim 的滚转、俯仰、真航向弧度转为 Dogfight 的 wxyz 四元数。"""
    cr, cp, cy = np.cos(rpy * 0.5)
    sr, sp, sy = np.sin(rpy * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    # 两侧都换基，世界 NED 到 Dogfight 为 (东, 上, 北)。
    # 换基矩阵行列式为 -1，四元数的旋转轴向量按 det(P) * P 变换。
    return np.array([w, -y, z, -x])


def slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    return (math.sin((1.0 - fraction) * angle) * first
            + math.sin(fraction * angle) * second) / math.sin(angle)


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray

    def matrix(self) -> list[float]:
        w, x, y, z = self.rotation
        # 服务端按列解包，最后一列为世界位置。
        return [
            1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
            2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
            2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y),
            *self.position.tolist(),
        ]

    def interpolate(self, other: Pose, fraction: float) -> Pose:
        return Pose(
            self.position + fraction * (other.position - self.position),
            slerp(self.rotation, other.rotation, fraction),
            self.velocity + fraction * (other.velocity - self.velocity),
        )


class Playback:
    def __init__(
        self, client: DogfightClient, plane_id: str, render_hz: int, origin_xz: tuple[float, float],
        log_file: TextIO,
    ):
        self.client = client
        self.plane_id = plane_id
        self.render_hz = render_hz
        self.origin_xz = origin_xz
        self.log_file = log_file
        self.writer = None
        self.reference_neu: np.ndarray | None = None
        self.last_pose: Pose | None = None
        self.last_time = 0.0
        self.clock_start = 0.0
        self.next_frame = 1
        self.next_status = 0.0
        self.last_stage = None
        self.send_ms: list[float] = []

    def _capture(self, env: heading.SingleControlEnv) -> Pose:
        sim = env.agents[heading.AGENT_ID]
        position = sim.get_position()
        if self.reference_neu is None:
            self.reference_neu = position.copy()
        north, east, _ = position - self.reference_neu
        x0, z0 = self.origin_xz
        pose = Pose(
            np.array([x0 + east, sim.get_geodetic()[2], z0 + north]),
            dogfight_quaternion(sim.get_rpy()),
            sim.get_velocity()[[1, 2, 0]],
        )
        if not all(np.isfinite(values).all() for values in (pose.position, pose.rotation, pose.velocity)):
            raise RuntimeError("JSBSim 位姿或速度包含非有限值")
        return pose

    def _send(self, pose: Pose) -> None:
        start = time.perf_counter()
        self.client.send_pose(self.plane_id, pose)
        self.send_ms.append(1000.0 * (time.perf_counter() - start))

    def __call__(self, env: heading.SingleControlEnv, record: dict | None) -> None:
        if record is None:
            pose = self._capture(env)
            self.client.use_external_pose(self.plane_id)
            self._send(pose)
            self.client.confirm_pose(self.plane_id, pose)
            self.last_pose = pose
            self.clock_start = time.perf_counter()
            return

        # 每步先落盘。即使网络发到一半中断，已完成的飞控记录也能保留。
        if self.writer is None:
            self.writer = csv.DictWriter(self.log_file, fieldnames=list(record))
            self.writer.writeheader()
        self.writer.writerow(record)
        self.log_file.flush()

        pose = self._capture(env)
        assert self.last_pose is not None
        simulation_time = record["time_sec"]
        last_due_frame = math.floor(simulation_time * self.render_hz + 1e-8)
        while self.next_frame <= last_due_frame:
            wall_deadline = self.clock_start + self.next_frame / self.render_hz
            delay = wall_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elapsed = time.perf_counter() - self.clock_start
            # 超时后跳到当前可显示的最新帧，仿真步长和策略推理频率保持固定。
            frame = min(last_due_frame, max(self.next_frame, math.floor(elapsed * self.render_hz)))
            fraction = (frame / self.render_hz - self.last_time) / (simulation_time - self.last_time)
            self._send(self.last_pose.interpolate(pose, min(1.0, max(0.0, fraction))))
            self.next_frame = frame + 1

        self.last_pose = pose
        self.last_time = simulation_time
        if simulation_time >= self.next_status or record["stage_index"] != self.last_stage:
            self.client.plane_state(self.plane_id)
            self._print_status(record)
            self.next_status = simulation_time + 1.0
            self.last_stage = record["stage_index"]

    @staticmethod
    def _print_status(record: dict) -> None:
        phase = STAGE_LABELS[record["stage_name"]]
        print(
            f'{record["time_sec"]:5.1f} s  {record["stage_index"]}/4 {phase}  '
            f'航向 {record["heading_deg"]:.1f}°/{record["target_heading_deg"]:.1f}°'
            f'（误差 {record["heading_error_deg"]:+.1f}°）  '
            f'高度 {record["altitude_m"]:.1f}/{record["target_altitude_m"]:.1f} m'
            f'（误差 {record["altitude_error_m"]:+.1f} m）  '
            f'前向速度 {record["speed_u_mps"]:.1f}/{record["target_speed_mps"]:.1f} m/s'
            f'（误差 {record["speed_error_mps"]:+.1f} m/s）',
            flush=True,
        )

    def finish(self) -> dict:
        assert self.last_pose is not None
        self._send(self.last_pose)
        state = self.client.confirm_pose(self.plane_id, self.last_pose)
        return {field: state[field] for field in (
            "position", "move_vector", "heading", "pitch_attitude", "roll_attitude",
        )}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, help="已通过离线验收的 Heading actor 权重")
    parser.add_argument("--host", help="Dogfight Network mode 显示的地址，默认本机主机名对应地址")
    parser.add_argument("--port", type=int, default=50888, help="Dogfight 的网络端口")
    parser.add_argument("--plane-id", help="Dogfight 内显示飞机的 ID，用 --list-planes 查询")
    parser.add_argument("--list-planes", action="store_true", help="查询当前 Network mode 中的飞机 ID")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 CPU")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render-hz", type=int, default=25, help="位姿发送频率，5–60 Hz，默认 25 Hz")
    parser.add_argument("--origin-xz", type=float, nargs=2, metavar=("X", "Z"), default=(0.0, 0.0),
                        help="Dogfight 场景中的水平起点，单位 m")
    parser.add_argument("--output-dir", type=Path, help="日志目录，默认在 experiments/results 下新建")
    args = parser.parse_args()
    if not args.list_planes and (args.model_path is None or args.plane_id is None):
        parser.error("演示需要同时传入 --model-path 和 --plane-id")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1–65535 之间")
    if not 5 <= args.render_hz <= 60:
        parser.error("--render-hz 必须在 5–60 之间")
    if not all(math.isfinite(value) for value in args.origin_xz):
        parser.error("--origin-xz 必须为有限值")
    return args


def run_demo(args: argparse.Namespace, client: DogfightClient) -> bool:
    planes = client.planes()
    if args.plane_id not in planes:
        raise ValueError(f"找不到飞机 {args.plane_id}，可用 ID 为 {', '.join(planes)}")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    actor = heading.build_actor(args.model_path, device)
    config_path = args.model_path.with_name("model_config.json")
    git = shutil.which("git")
    revision = None
    if git is not None:
        revision = subprocess.run(
            [git, "describe", "--always", "--dirty"], cwd=heading.REPO_ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or None
    output = args.output_dir or (
        heading.REPO_ROOT / "experiments" / "results" / "dogfight_heading"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": args.model_path.name,
        "model_sha256": hashlib.sha256(args.model_path.read_bytes()).hexdigest(),
        "model_config": json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None,
        "memory_encoder": actor.memory_encoder,
        "state_shape": list(actor.state_shape),
        "demo2_revision": revision,
        "device": str(device),
        "seed": args.seed,
        "scenario": heading.SCENARIO_NAME,
        "dogfight_plane_id": args.plane_id,
        "render_hz": args.render_hz,
        "origin_xz_m": args.origin_xz,
    }
    print(f"正在显示 {args.plane_id}。下列数值依次为实际值 / 目标值 / 误差。", flush=True)
    print(f"演示日志写入 {output.resolve()}", flush=True)
    env = heading.SingleControlEnv(heading.SCENARIO_NAME)
    playback = None
    try:
        horizon = heading.fixed_sequence_horizon_steps(env)
        metadata["sequence"] = heading.fixed_sequence_config(horizon, env.time_interval)
        with (output / "timeseries.csv").open("x", newline="", encoding="utf-8") as log_file:
            playback = Playback(client, args.plane_id, args.render_hz, args.origin_xz, log_file)
            episode, _, stages = heading.run_trial(
                env, actor, trial_id=1, seed=args.seed, command=None,
                horizon_steps=horizon, device=device, on_snapshot=playback,
            )
        heading.write_csv(output / "episodes.csv", [episode])
        heading.write_csv(output / "stages.csv", stages)
        metadata["tracking_passed"] = episode["success"]
        metadata["final_dogfight_state"] = playback.finish()
        metadata["status"] = "completed"
        print(f"联机结束，完成 {episode['completed_stages']}/4 阶段，"
              f"跟踪验收{'通过' if episode['success'] else '未通过'}。", flush=True)
        return episode["success"]
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="interrupted", error=str(error) or type(error).__name__)
        raise
    finally:
        env.close()
        if playback is not None:
            metadata["last_displayed_sim_time_sec"] = playback.last_time
            metadata["pose_packets_sent"] = len(playback.send_ms)
            if playback.send_ms:
                metadata["send_duration_p95_ms"] = float(np.percentile(playback.send_ms, 95))
            if playback.clock_start:
                elapsed = time.perf_counter() - playback.clock_start
                metadata["elapsed_wall_sec"] = elapsed
                metadata["effective_pose_hz"] = len(playback.send_ms) / elapsed
            (output / "summary.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )


def main() -> int:
    # Git Bash 的终端按 UTF-8 解码，Windows Python 连到其管道时可能默认用系统代码页。
    if sys.platform == "win32" and os.environ.get("MSYSTEM") and "PYTHONIOENCODING" not in os.environ:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    host = args.host or "本机"
    try:
        host = args.host or socket.gethostbyname(socket.gethostname())
        try:
            connection = socket.create_connection((host, args.port), timeout=SOCKET_TIMEOUT_SEC)
        except OSError as error:
            raise ConnectionError(f"无法连接到 Network mode。请核对窗口左上角的地址和端口。{error}") from error
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client = DogfightClient(connection)
            try:
                client.command("DISABLE_LOG")
                if args.list_planes:
                    print("\n".join(client.planes()))
                    return 0
                return 0 if run_demo(args, client) else 1
            finally:
                client.disconnect()
    except KeyboardInterrupt:
        print("演示已中断。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Dogfight 演示失败（{host}:{args.port}）\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
