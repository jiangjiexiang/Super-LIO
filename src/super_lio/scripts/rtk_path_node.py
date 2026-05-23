#!/usr/bin/env python3

import math
import re

import rospy
import rostopic
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3

# GPGGA fix quality: 0=invalid, 1=GPS, 2=DGPS, 4=RTK fixed, 5=RTK float
GGA_FIX_RE = re.compile(r"\$G[PN]GGA,")


class RtkPathNode:
    def __init__(self):
        self.rtk_topic = rospy.get_param("~rtk_topic", "/rtk/fix")
        self.path_topic = rospy.get_param("~path_topic", "/rtk/path")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.relative = rospy.get_param("~relative", True)
        self.use_altitude = rospy.get_param("~use_altitude", False)
        self.max_path_length = int(rospy.get_param("~max_path_length", 0))
        yaw_offset_deg = float(rospy.get_param("~yaw_offset_deg", 0.0))
        self.yaw_offset = math.radians(yaw_offset_deg)
        self._cos_yaw = 1.0
        self._sin_yaw = 0.0
        self._update_yaw_cache()
        self.extra_yaw_offset_deg = float(rospy.get_param("~extra_yaw_offset_deg", 0.0))
        self.auto_align_yaw = bool(rospy.get_param("~auto_align_yaw", False))
        self.align_min_dist = float(rospy.get_param("~align_min_dist", 3.0))
        self.motion_yaw = None
        self.yaw_aligned = not self.auto_align_yaw
        self.pending = []
        self.lio_pos = None
        self.compare_every_n = int(rospy.get_param("~compare_every_n", 100))
        self.compare_count = 0
        self.path_publish_hz = float(rospy.get_param("~path_publish_hz", 30.0))
        self.path_dirty = False
        rtk_queue = int(rospy.get_param("~rtk_queue_size", 2000))
        if rtk_queue <= 0:
            rtk_queue = 2000

        self.gnss_raw_topic = rospy.get_param("~gnss_raw_topic", "/apollo/sensor/gnss/raw_data")
        self.require_gnss_fix = bool(rospy.get_param("~require_gnss_fix", True))
        self.min_fix_quality = int(rospy.get_param("~min_fix_quality", 4))
        self.require_rtk_fixed = bool(rospy.get_param("~require_rtk_fixed", True))
        self.gnss_max_age = float(rospy.get_param("~gnss_max_age", 2.0))
        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 12.0))
        self.max_jump_xy = float(rospy.get_param("~max_jump_xy", 3.0))
        self.resume_jump_xy = float(rospy.get_param("~resume_jump_xy", 80.0))
        self.gap_resume_sec = float(rospy.get_param("~gap_resume_sec", 8.0))
        self.gap_resume_until = None
        self.gnss_fix = None
        self.gnss_fix_stamp = None
        self.last_good_xy = None
        self.last_good_stamp = None
        self.gnss_ok = True
        self._warned_degraded = False

        self.path = Path()
        self.path.header.frame_id = self.frame_id
        self.origin_xyz = None
        self.origin_enu = None
        self.origin_lat = None

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)

        if self.require_gnss_fix and self.gnss_raw_topic:
            rospy.Subscriber(self.gnss_raw_topic, String, self.gnss_raw_callback, queue_size=50)
            rule = "fix==4 only" if self.require_rtk_fixed else "fix>=%d" % self.min_fix_quality
            rospy.loginfo(
                "RTK path: GPGGA from %s (%s); indoor non-fixed pauses green path",
                self.gnss_raw_topic,
                rule,
            )

        msg_class, _, _ = rostopic.get_topic_class(self.rtk_topic, blocking=True)
        if msg_class is None:
            raise RuntimeError("Cannot resolve RTK topic type: {}".format(self.rtk_topic))

        sub_kw = {"queue_size": rtk_queue}
        if msg_class == NavSatFix:
            self.sub = rospy.Subscriber(self.rtk_topic, NavSatFix, self.navsat_callback, **sub_kw)
        elif msg_class == Odometry:
            self.sub = rospy.Subscriber(self.rtk_topic, Odometry, self.odom_callback, **sub_kw)
        elif msg_class == PoseStamped:
            self.sub = rospy.Subscriber(self.rtk_topic, PoseStamped, self.pose_callback, **sub_kw)
        else:
            raise RuntimeError(
                "Unsupported RTK topic type {} on {}. Use sensor_msgs/NavSatFix, "
                "nav_msgs/Odometry, or geometry_msgs/PoseStamped.".format(msg_class._type, self.rtk_topic)
            )

        self.lio_topic = rospy.get_param("~lio_odom_topic", "/lio/odom")
        rospy.Subscriber(self.lio_topic, Odometry, self.lio_odom_callback, queue_size=10)

        if self.path_publish_hz > 0.0:
            rospy.Timer(rospy.Duration(1.0 / self.path_publish_hz), self._publish_path_timer)
        else:
            rospy.logwarn("path_publish_hz<=0: publish full Path on every RTK (slow for long paths)")

        rospy.loginfo("RTK path: %s (%s) -> %s", self.rtk_topic, msg_class._type, self.path_topic)
        rospy.loginfo(
            "RTK path: append when GNSS OK, publish at %.1f Hz; jump gate %.1f m / %.1f m/s",
            self.path_publish_hz,
            self.max_jump_xy,
            self.max_speed_mps,
        )
        rospy.loginfo("Path compare: %s vs %s, log every %d RTK samples", self.lio_topic, self.path_topic, self.compare_every_n)
        if self.auto_align_yaw:
            rospy.loginfo(
                "RTK path: wait %.1f m travel before publishing (avoid pre-align wrong direction)",
                self.align_min_dist,
            )

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
            self.last_good_xy = None
            self.last_good_stamp = None
            self.gap_resume_until = None
            self._set_gnss_ok(False)
        elif prev is not None and prev != 4 and fix == 4:
            self.last_good_xy = None
            self.last_good_stamp = None
            self.gap_resume_until = rospy.Time.now() + rospy.Duration(self.gap_resume_sec)
            rospy.loginfo(
                "RTK path: outdoor RTK fixed — resume in %.0fs (allow jump<=%.0f m to reconnect)",
                self.gap_resume_sec,
                self.resume_jump_xy,
            )
            self._warned_degraded = False

    def _gnss_quality_fresh(self):
        """GPGGA 是否在有效期内（用 now 判断，避免 RTK 旧时间戳误判 stale）。"""
        if self.gnss_fix is None or self.gnss_fix_stamp is None:
            return False
        return (rospy.Time.now() - self.gnss_fix_stamp).to_sec() <= self.gnss_max_age

    def _gnss_fix_ok(self, _stamp=None):
        if not self.require_gnss_fix:
            return True
        if not self._gnss_quality_fresh():
            return False
        if self.require_rtk_fixed:
            return self.gnss_fix == 4
        return self.gnss_fix >= self.min_fix_quality

    def _set_gnss_ok(self, ok):
        if ok == self.gnss_ok:
            return
        self.gnss_ok = ok
        if ok:
            rospy.loginfo("RTK path: GNSS fix recovered (fix=%s), resume drawing", str(self.gnss_fix))
            self._warned_degraded = False
            self.last_good_xy = None
            self.last_good_stamp = None
        elif not self._warned_degraded:
            if self.gnss_fix is None:
                rospy.logwarn_throttle(5.0, "RTK path: waiting for GPGGA on %s", self.gnss_raw_topic)
            else:
                need = "fix==4" if self.require_rtk_fixed else "fix>=%d" % self.min_fix_quality
                rospy.logwarn(
                    "RTK path: GNSS degraded (fix=%s, need %s) — pause green path (indoor)",
                    str(self.gnss_fix),
                    need,
                )
            self._warned_degraded = True

    def _max_jump_allowed(self):
        if self.gap_resume_until is not None and rospy.Time.now() <= self.gap_resume_until:
            return self.resume_jump_xy
        return self.max_jump_xy

    def _last_path_xy(self):
        if not self.path.poses:
            return None
        p = self.path.poses[-1].pose.position
        return (p.x, p.y)

    def _motion_ok(self, stamp, x, y):
        if self.last_good_xy is None:
            return True
        step = math.hypot(x - self.last_good_xy[0], y - self.last_good_xy[1])
        if step > self._max_jump_allowed():
            return False
        if self.last_good_stamp is not None and stamp is not None and stamp != rospy.Time():
            dt = (stamp - self.last_good_stamp).to_sec()
            if dt > 1e-4:
                if step > self.max_speed_mps * dt + 0.15:
                    return False
        return True

    def _accept_sample(self, stamp, x, y):
        if stamp is None or stamp == rospy.Time():
            stamp = rospy.Time.now()

        if not self._gnss_fix_ok(stamp):
            self._set_gnss_ok(False)
            return False

        if not self._motion_ok(stamp, x, y):
            rospy.logwarn_throttle(
                5.0,
                "RTK path: skip sample jump=%.1f m (limit %.1f)",
                math.hypot(x - self.last_good_xy[0], y - self.last_good_xy[1]),
                self._max_jump_allowed(),
            )
            return False

        self._set_gnss_ok(True)
        self.last_good_xy = (x, y)
        self.last_good_stamp = stamp
        return True

    def _update_yaw_cache(self):
        self._cos_yaw = math.cos(self.yaw_offset)
        self._sin_yaw = math.sin(self.yaw_offset)

    def _publish_path_timer(self, _event):
        if not self.path_dirty:
            return
        self._publish_path()

    def _publish_path(self):
        if not self.path.poses:
            return
        self.path.header.stamp = self.path.poses[-1].header.stamp
        self.pub_path.publish(self.path)
        self.path_dirty = False

    def navsat_callback(self, msg):
        if msg.status.status < 0:
            return
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        if self.require_gnss_fix and msg.status.status < self.min_fix_quality:
            self.gnss_fix = msg.status.status
            self.gnss_fix_stamp = msg.header.stamp
            self._set_gnss_ok(False)
            return

        xyz = self.geodetic_to_ecef(msg.latitude, msg.longitude, msg.altitude if self.use_altitude else 0.0)
        if self.origin_xyz is None:
            self.origin_xyz = xyz
            self.origin_lat = math.radians(msg.latitude)
            self.origin_lon = math.radians(msg.longitude)

        position = self.ecef_to_enu(xyz, self.origin_xyz, self.origin_lat, self.origin_lon)
        self.append_pose(msg.header.stamp, position)

    def lio_odom_callback(self, msg):
        p = msg.pose.pose.position
        self.lio_pos = (p.x, p.y, p.z)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        self.append_pose(msg.header.stamp, (position.x, position.y, position.z))

    def pose_callback(self, msg):
        position = msg.pose.position
        self.append_pose(msg.header.stamp, (position.x, position.y, position.z))

    def _relative_xyz(self, position):
        x, y, z = position
        if self.relative:
            if self.origin_enu is None:
                self.origin_enu = (x, y, z)
            x -= self.origin_enu[0]
            y -= self.origin_enu[1]
            z -= self.origin_enu[2]
        return x, y, z

    def _rotate_xy(self, x, y):
        if self.yaw_offset == 0.0:
            return x, y
        return self._cos_yaw * x - self._sin_yaw * y, self._sin_yaw * x + self._cos_yaw * y

    def _try_auto_align(self, x, y):
        if not self.auto_align_yaw or self.yaw_aligned:
            return
        dist_xy = math.hypot(x, y)
        if dist_xy < self.align_min_dist:
            return
        self.motion_yaw = math.atan2(y, x)
        self.yaw_offset = -self.motion_yaw + math.radians(self.extra_yaw_offset_deg)
        self._update_yaw_cache()
        self.yaw_aligned = True
        rospy.loginfo(
            "RTK path auto yaw: motion=%.1f deg, offset=%.1f deg, flush %d buffered poses",
            math.degrees(self.motion_yaw),
            math.degrees(self.yaw_offset),
            len(self.pending),
        )
        for stamp, px, py, pz in self.pending:
            if self._gnss_fix_ok(stamp) and self._motion_ok(stamp, px, py):
                self._push_pose(stamp, px, py, pz)
        self.pending = []
        self._publish_path()

    def _push_pose(self, stamp, x, y, z):
        x, y = self._rotate_xy(x, y)
        last_xy = self._last_path_xy()
        if last_xy is not None:
            gap = math.hypot(x - last_xy[0], y - last_xy[1])
            if gap > self.max_jump_xy:
                rospy.loginfo_throttle(
                    3.0,
                    "RTK path: continue after %.1f m gap (indoor break; RViz may show a long chord)",
                    gap,
                )
        pose = PoseStamped()
        pose.header.stamp = stamp if stamp != rospy.Time() else rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self.path.poses.append(pose)
        if self.max_path_length > 0 and len(self.path.poses) > self.max_path_length:
            self.path.poses = self.path.poses[-self.max_path_length :]
        self.path_dirty = True

    def _log_path_compare(self, rtk_x, rtk_y, rtk_z):
        if self.lio_pos is None:
            return
        self.compare_count += 1
        if self.compare_count % self.compare_every_n != 0:
            return
        dx = self.lio_pos[0] - rtk_x
        dy = self.lio_pos[1] - rtk_y
        dz = self.lio_pos[2] - rtk_z
        d_xy = math.hypot(dx, dy)
        d_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        rospy.loginfo(
            "Compare LIO path vs RTK path | d_xy: %.3f m, d_z: %.3f m, d_3d: %.3f m",
            d_xy,
            abs(dz),
            d_3d,
        )

    def append_pose(self, stamp, position):
        x, y, z = self._relative_xyz(position)

        if self.auto_align_yaw and not self.yaw_aligned:
            self.pending.append((stamp, x, y, z))
            self._try_auto_align(x, y)
            return

        if not self._accept_sample(stamp, x, y):
            return

        rx, ry = self._rotate_xy(x, y)
        self._log_path_compare(rx, ry, z)
        self._push_pose(stamp, x, y, z)
        if self.path_publish_hz <= 0.0:
            self._publish_path()

    @staticmethod
    def geodetic_to_ecef(lat_deg, lon_deg, alt):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        x = (n + alt) * cos_lat * math.cos(lon)
        y = (n + alt) * cos_lat * math.sin(lon)
        z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
        return x, y, z

    @staticmethod
    def ecef_to_enu(xyz, origin_xyz, origin_lat, origin_lon):
        dx = xyz[0] - origin_xyz[0]
        dy = xyz[1] - origin_xyz[1]
        dz = xyz[2] - origin_xyz[2]

        sin_lat = math.sin(origin_lat)
        cos_lat = math.cos(origin_lat)
        sin_lon = math.sin(origin_lon)
        cos_lon = math.cos(origin_lon)

        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        return east, north, up


if __name__ == "__main__":
    rospy.init_node("rtk_path_node")
    RtkPathNode()
    rospy.spin()
