#!/usr/bin/env python3
"""
Loose coupling: 3D position EKF fusing LIO increments + RTK position (aligned to LIO world).
Outdoor RTK pulls drift; indoor (fix<4) prediction follows LIO only.
"""

import math
import re

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String

GGA_FIX_RE = re.compile(r"\$G[PN]GGA,")


class PosEkf3(object):
    def __init__(self, q_lio, r_rtk_xy, r_rtk_z):
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64)
        self.Q = np.diag([q_lio * q_lio, q_lio * q_lio, q_lio * q_lio * 0.5])
        self.R = np.diag([r_rtk_xy * r_rtk_xy, r_rtk_xy * r_rtk_xy, r_rtk_z * r_rtk_z])
        self.ready = False

    def reset(self, p):
        self.x = np.asarray(p, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 0.25
        self.ready = True

    def predict_delta(self, dp):
        self.x += np.asarray(dp, dtype=np.float64)
        self.P += self.Q

    def update(self, z, R=None, max_innovation=None):
        z = np.asarray(z, dtype=np.float64)
        Rm = self.R if R is None else R
        y = z - self.x
        innov = float(np.linalg.norm(y))
        if max_innovation is not None and innov > max_innovation:
            return innov, False
        S = self.P + Rm
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K) @ self.P
        return innov, True


class LioRtkEkfNode(object):
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.lio_topic = rospy.get_param("~lio_odom_topic", "/lio/odom")
        self.rtk_topic = rospy.get_param("~rtk_topic", "/apollo/localization/ins570d/pose")
        self.gnss_raw_topic = rospy.get_param("~gnss_raw_topic", "/apollo/sensor/gnss/raw_data")
        self.path_topic = rospy.get_param("~path_topic", "/fusion/path")
        self.odom_topic = rospy.get_param("~odom_topic", "/fusion/odom")

        self.auto_align_yaw = bool(rospy.get_param("~auto_align_yaw", True))
        self.align_min_dist = float(rospy.get_param("~align_min_dist", 3.0))
        self.extra_yaw_offset_deg = float(rospy.get_param("~extra_yaw_offset_deg", 0.0))
        self.yaw_offset = math.radians(float(rospy.get_param("~yaw_offset_deg", 0.0)))
        self._update_yaw_cache()
        self.yaw_aligned = not self.auto_align_yaw
        self.rtk_origin = None
        self.rtk_pending = []

        self.min_fix_quality = int(rospy.get_param("~min_fix_quality", 4))
        # 默认仅 fix=4（RTK 固定解）；室内 fix=2/5 等一律不融合
        self.require_rtk_fixed = bool(rospy.get_param("~require_rtk_fixed", True))
        self.gnss_max_age = float(rospy.get_param("~gnss_max_age", 2.0))
        self.rtk_sync_max_dt = float(rospy.get_param("~rtk_sync_max_dt", 0.08))
        self.max_innovation = float(rospy.get_param("~max_innovation", 8.0))
        self.max_rtk_jump_xy = float(rospy.get_param("~max_rtk_jump_xy", 3.0))
        self.resume_rtk_jump_xy = float(rospy.get_param("~resume_rtk_jump_xy", 80.0))
        self.gap_resume_sec = float(rospy.get_param("~gap_resume_sec", 8.0))
        self.gap_resume_until = None

        q_lio = float(rospy.get_param("~sigma_lio", 0.08))
        self.sigma_rtk_fixed = float(rospy.get_param("~sigma_rtk", 0.05))
        self.sigma_rtk_float = float(rospy.get_param("~sigma_rtk_float", 0.35))
        r_rtk_z = float(rospy.get_param("~sigma_rtk_z", 0.12))
        self.ekf = PosEkf3(q_lio, self.sigma_rtk_fixed, r_rtk_z)

        self.gnss_fix = None
        self.gnss_fix_stamp = None
        self.latest_rtk = None  # (stamp, np.array3)
        self.last_rtk_xy = None
        self.fusion_mode = "init"  # init | lio_only | fused
        self._warned_mode = False
        self.last_lio = None
        self.last_lio_stamp = None

        self.path = Path()
        self.path.header.frame_id = self.frame_id
        self.max_path_length = int(rospy.get_param("~max_path_length", 0))
        self.path_publish_hz = float(rospy.get_param("~path_publish_hz", 30.0))
        self.path_dirty = False

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.pub_odom = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)

        rospy.Subscriber(self.lio_topic, Odometry, self.lio_callback, queue_size=50)
        rospy.Subscriber(self.rtk_topic, Odometry, self.rtk_callback, queue_size=200)
        if self.gnss_raw_topic:
            rospy.Subscriber(self.gnss_raw_topic, String, self.gnss_raw_callback, queue_size=50)

        if self.path_publish_hz > 0.0:
            rospy.Timer(rospy.Duration(1.0 / self.path_publish_hz), self._publish_path_timer)

        fix_rule = "fix==4 only" if self.require_rtk_fixed else "fix>=%d" % self.min_fix_quality
        rospy.loginfo(
            "LIO+RTK EKF: %s -> %s | RTK when %s, else LIO-only (indoor)",
            self.lio_topic,
            self.path_topic,
            fix_rule,
        )

    def _update_yaw_cache(self):
        self._cos_yaw = math.cos(self.yaw_offset)
        self._sin_yaw = math.sin(self.yaw_offset)

    @staticmethod
    def _parse_gpgga_fix(data):
        if not data:
            return None
        text = data if isinstance(data, str) else data.decode("utf-8", errors="ignore")
        for line in text.replace("\r", "\n").split("\n"):
            if not GGA_FIX_RE.search(line):
                continue
            parts = line.strip().split(",")
            if len(parts) < 7 or not parts[6].isdigit():
                continue
            return int(parts[6])
        return None

    def gnss_raw_callback(self, msg):
        fix = self._parse_gpgga_fix(msg.data)
        if fix is None:
            return
        prev = self.gnss_fix
        self.gnss_fix = fix
        # std_msgs/String 无 header，用接收时刻（与 /clock 一致）
        self.gnss_fix_stamp = rospy.Time.now()
        if prev == 4 and fix != 4:
            self._drop_rtk_buffer("GPGGA fix %d (indoor / non-RTK-fixed)" % fix)
        elif prev is not None and prev != 4 and fix == 4:
            self.last_rtk_xy = None
            self.gap_resume_until = rospy.Time.now() + rospy.Duration(self.gap_resume_sec)
            rospy.loginfo(
                "Fusion EKF: RTK fixed — resume %.0fs, relaxed jump %.0f m",
                self.gap_resume_sec,
                self.resume_rtk_jump_xy,
            )

    def _drop_rtk_buffer(self, reason):
        self.latest_rtk = None
        self.last_rtk_xy = None
        self.gap_resume_until = None
        self._set_fusion_mode("lio_only", rospy.Time.now(), reason)

    def _max_rtk_jump_allowed(self):
        if self.gap_resume_until is not None and rospy.Time.now() <= self.gap_resume_until:
            return self.resume_rtk_jump_xy
        return self.max_rtk_jump_xy

    def _gnss_quality_fresh(self):
        if self.gnss_fix is None or self.gnss_fix_stamp is None:
            return False
        return (rospy.Time.now() - self.gnss_fix_stamp).to_sec() <= self.gnss_max_age

    def _gnss_ok(self, _stamp=None):
        if not self._gnss_quality_fresh():
            return False
        if self.require_rtk_fixed:
            return self.gnss_fix == 4
        return self.gnss_fix >= self.min_fix_quality

    def _gnss_degraded_reason(self, _stamp=None):
        if self.gnss_fix is None:
            return "no GPGGA yet"
        if not self._gnss_quality_fresh():
            return "GPGGA expired"
        if self.require_rtk_fixed and self.gnss_fix != 4:
            return "fix=%d need RTK fixed(4)" % self.gnss_fix
        if not self.require_rtk_fixed and self.gnss_fix < self.min_fix_quality:
            return "fix=%d need>=%d" % (self.gnss_fix, self.min_fix_quality)
        return None

    def _rtk_R(self):
        s = self.sigma_rtk_fixed if self.gnss_fix == 4 else self.sigma_rtk_float
        return np.diag([s * s, s * s, self.ekf.R[2, 2]])

    def _set_fusion_mode(self, mode, stamp, reason=None):
        if mode == self.fusion_mode:
            return
        prev = self.fusion_mode
        self.fusion_mode = mode
        if mode == "fused":
            rospy.loginfo(
                "Fusion EKF: RTK fused (fix=%s, sigma=%.2f)",
                str(self.gnss_fix),
                self._rtk_R()[0, 0] ** 0.5,
            )
            self._warned_mode = False
        elif mode == "lio_only" and prev in ("fused", "init"):
            rospy.logwarn(
                "Fusion EKF: LIO-only — %s",
                reason or "RTK degraded",
            )
            self._warned_mode = True

    def _rtk_relative(self, p):
        if self.rtk_origin is None:
            self.rtk_origin = (p.x, p.y, p.z)
        return (
            p.x - self.rtk_origin[0],
            p.y - self.rtk_origin[1],
            p.z - self.rtk_origin[2],
        )

    def _rotate_xy(self, x, y):
        if self.yaw_offset == 0.0:
            return x, y
        return (
            self._cos_yaw * x - self._sin_yaw * y,
            self._sin_yaw * x + self._cos_yaw * y,
        )

    def _rtk_to_lio(self, p, stamp):
        x, y, z = self._rtk_relative(p)
        if self.auto_align_yaw and not self.yaw_aligned:
            self.rtk_pending.append((stamp, x, y, z))
            dist = math.hypot(x, y)
            if dist >= self.align_min_dist:
                motion_yaw = math.atan2(y, x)
                self.yaw_offset = -motion_yaw + math.radians(self.extra_yaw_offset_deg)
                self._update_yaw_cache()
                self.yaw_aligned = True
                rospy.loginfo(
                    "Fusion EKF yaw align: motion=%.1f deg offset=%.1f deg",
                    math.degrees(motion_yaw),
                    math.degrees(self.yaw_offset),
                )
            else:
                return None
        x, y = self._rotate_xy(x, y)
        return np.array([x, y, z], dtype=np.float64)

    def rtk_callback(self, msg):
        stamp = msg.header.stamp
        z = self._rtk_to_lio(msg.pose.pose.position, stamp)
        if z is None:
            return
        if not self.yaw_aligned:
            return
        if not self._gnss_ok(stamp):
            return
        xy = (float(z[0]), float(z[1]))
        if self.last_rtk_xy is not None:
            jump = math.hypot(xy[0] - self.last_rtk_xy[0], xy[1] - self.last_rtk_xy[1])
            if jump > self._max_rtk_jump_allowed():
                rospy.logwarn_throttle(
                    10.0,
                    "Fusion EKF: skip RTK jump=%.1f m (limit %.1f)",
                    jump,
                    self._max_rtk_jump_allowed(),
                )
                return
        self.last_rtk_xy = xy
        self.latest_rtk = (stamp, z)

    def _try_rtk_update(self, lio_stamp):
        if self.latest_rtk is None:
            self._set_fusion_mode("lio_only", lio_stamp, "no RTK synced")
            return
        rtk_stamp, z = self.latest_rtk
        dt = abs((lio_stamp - rtk_stamp).to_sec())
        if dt > self.rtk_sync_max_dt:
            self._set_fusion_mode("lio_only", lio_stamp, "RTK time sync fail")
            return
        if not self._gnss_ok(lio_stamp):
            bad = self._gnss_degraded_reason(lio_stamp)
            self._set_fusion_mode("lio_only", lio_stamp, bad)
            self.latest_rtk = None
            return
        R = self._rtk_R()
        innov, ok = self.ekf.update(z, R=R, max_innovation=self.max_innovation)
        if not ok:
            rospy.logwarn_throttle(
                2.0,
                "Fusion EKF: reject RTK (innov=%.2f m > %.2f), stay LIO",
                innov,
                self.max_innovation,
            )
            self._set_fusion_mode("lio_only", lio_stamp, "innovation gate")
        else:
            self._set_fusion_mode("fused", lio_stamp)
        self.latest_rtk = None

    def lio_callback(self, msg):
        if not self.yaw_aligned:
            return

        p = msg.pose.pose.position
        lio = np.array([p.x, p.y, p.z], dtype=np.float64)
        stamp = msg.header.stamp

        if not self.ekf.ready:
            self.ekf.reset(lio)
            self.last_lio = lio.copy()
            self.last_lio_stamp = stamp
            self._publish_fusion(stamp, msg.pose.pose.orientation)
            return

        if self.last_lio is not None:
            dp = lio - self.last_lio
            self.ekf.predict_delta(dp)

        self._try_rtk_update(stamp)

        self.last_lio = lio.copy()
        self.last_lio_stamp = stamp
        self._publish_fusion(stamp, msg.pose.pose.orientation)

    def _publish_fusion(self, stamp, orient):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = "fusion"
        odom.pose.pose.position.x = self.ekf.x[0]
        odom.pose.pose.position.y = self.ekf.x[1]
        odom.pose.pose.position.z = self.ekf.x[2]
        odom.pose.pose.orientation = orient
        self.pub_odom.publish(odom)

        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self.path.poses.append(ps)
        if self.max_path_length > 0 and len(self.path.poses) > self.max_path_length:
            self.path.poses = self.path.poses[-self.max_path_length :]
        self.path_dirty = True
        if self.path_publish_hz <= 0.0:
            self._publish_path()

    def _publish_path_timer(self, _evt):
        if self.path_dirty:
            self._publish_path()

    def _publish_path(self):
        if not self.path.poses:
            return
        self.path.header.stamp = self.path.poses[-1].header.stamp
        self.pub_path.publish(self.path)
        self.path_dirty = False


if __name__ == "__main__":
    rospy.init_node("lio_rtk_ekf_node")
    LioRtkEkfNode()
    rospy.spin()
