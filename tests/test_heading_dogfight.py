import json
import math

import numpy as np
import pytest

from experiments.run_heading_dogfight import DogfightClient, Pose, dogfight_quaternion


@pytest.mark.parametrize(("rpy_deg", "axis", "expected"), [
    ((0, 0, 0), 2, (0, 0, 1)),
    ((0, 0, 90), 2, (1, 0, 0)),
    ((0, 30, 0), 2, (0, 0.5, math.sqrt(0.75))),
    ((30, 0, 0), 0, (math.sqrt(0.75), -0.5, 0)),
])
def test_dogfight_matrix_uses_local_axes_as_columns(rpy_deg, axis, expected):
    pose = Pose(np.array([1, 2, 3]), dogfight_quaternion(np.deg2rad(rpy_deg)), np.zeros(3))
    packet = pose.matrix()
    np.testing.assert_allclose(packet[axis * 3:axis * 3 + 3], expected, atol=1e-12)
    assert packet[9:] == [1, 2, 3]


def test_dogfight_pose_interpolation_crosses_north_without_a_full_turn():
    first = Pose(np.zeros(3), dogfight_quaternion(np.deg2rad([0, 0, 350])), np.zeros(3))
    second = Pose(np.array([2, 4, 6]), dogfight_quaternion(np.deg2rad([0, 0, 10])), np.ones(3))
    midpoint = first.interpolate(second, 0.5)
    np.testing.assert_allclose(midpoint.matrix()[6:9], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(midpoint.position, [1, 2, 3])
    np.testing.assert_allclose(midpoint.velocity, [0.5, 0.5, 0.5])


class FragmentedSocket:
    def __init__(self, reply):
        self.reply = reply
        self.request = b""

    def sendall(self, data):
        self.request += data

    def recv(self, size):
        # 模拟 TCP 将长度头和消息体拆成多个片段。
        length = min(size, 2)
        fragment, self.reply = self.reply[:length], self.reply[length:]
        return fragment


def test_dogfight_client_reads_fragmented_json_reply():
    reply = b'["ally_2"]'
    connection = FragmentedSocket(len(reply).to_bytes(4, "big") + reply)
    assert DogfightClient(connection).planes() == ["ally_2"]
    header, request = connection.request[:4], connection.request[4:]
    assert int.from_bytes(header, "big") == len(request)
    assert json.loads(request) == {"command": "GET_PLANESLIST", "args": {}}


def test_dogfight_disconnect_mid_reply_reports_connection_error():
    connection = FragmentedSocket((12).to_bytes(4, "big") + b'["')
    with pytest.raises(ConnectionError, match="断开连接"):
        DogfightClient(connection).planes()


def test_dogfight_disconnect_sends_server_close_marker():
    connection = FragmentedSocket(b"")
    DogfightClient(connection).disconnect()
    assert int.from_bytes(connection.request[:4], "big") == len(connection.request[4:])
    assert json.loads(connection.request[4:]) == ""


def test_dogfight_timeout_identifies_command():
    class UnresponsiveSocket(FragmentedSocket):
        def recv(self, size):
            raise TimeoutError("timed out")

    with pytest.raises(TimeoutError, match="GET_PLANESLIST"):
        DogfightClient(UnresponsiveSocket(b"")).planes()
