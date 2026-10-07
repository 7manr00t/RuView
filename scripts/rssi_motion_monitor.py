#!/usr/bin/env python3
"""Live WiFi RSSI motion monitor for a laptop's own WiFi adapter.

Samples the RSSI of the access point the laptop is connected to several times
a second, plots it live, and flags MOTION when the rolling variance exceeds a
threshold calibrated on an empty room.

This is coarse RSSI-only sensing, not CSI and not camera-grade. It can detect
that something moved near the radio path; it cannot localise, count, or pose
people. Detection quality depends heavily on the driver's RSSI refresh rate
(many drivers update only once per beacon interval or slower; Windows reports
signal quality in whole percent).

It never generates or simulates data: if no real adapter reading is available
it exits with an explanation.

Usage:
    python scripts/rssi_motion_monitor.py               # calibrate 30 s, then monitor
    python scripts/rssi_motion_monitor.py --no-plot     # console only
    python scripts/rssi_motion_monitor.py --interface wlp2s0 --rate 8 --csv out.csv

Optional dependency for the live plot: matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Deque, List, Optional, Tuple


class RssiUnavailable(RuntimeError):
    """Raised when no real RSSI reading can be obtained from this host."""


# ---------------------------------------------------------------------------
# Platform RSSI sources (real hardware only)
# ---------------------------------------------------------------------------


class LinuxRssi:
    """Reads RSSI from /proc/net/wireless, falling back to `iw dev <if> link`."""

    def __init__(self, interface: Optional[str] = None):
        self.interface = interface or self._detect_interface()
        self.name = f"linux:{self.interface}"

    @staticmethod
    def _wireless_interfaces() -> List[str]:
        base = "/sys/class/net"
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return []
        return [n for n in names if os.path.isdir(os.path.join(base, n, "wireless"))]

    def _detect_interface(self) -> str:
        found = self._wireless_interfaces()
        if not found:
            raise RssiUnavailable(
                "No wireless interface found under /sys/class/net (only: "
                + ", ".join(sorted(os.listdir("/sys/class/net")) if os.path.isdir("/sys/class/net") else [])
                + "). This host has no WiFi adapter, or it is inside a VM/container "
                "that cannot see the host's radio."
            )
        return found[0]

    def _from_proc(self) -> Optional[float]:
        try:
            with open("/proc/net/wireless", encoding="ascii") as fh:
                lines = fh.readlines()[2:]
        except OSError:
            return None
        for line in lines:
            if line.split(":", 1)[0].strip() == self.interface:
                fields = line.split(":", 1)[1].split()
                # status, link quality, level (dBm), noise, ...
                level = float(fields[2].rstrip("."))
                if level > 0:  # some drivers report unsigned 8-bit values
                    level -= 256
                return level if level < 0 else None
        return None

    def _from_iw(self) -> Optional[float]:
        if not shutil.which("iw"):
            return None
        out = subprocess.run(
            ["iw", "dev", self.interface, "link"], capture_output=True, text=True, timeout=2
        ).stdout
        m = re.search(r"signal:\s*(-?\d+)\s*dBm", out)
        return float(m.group(1)) if m else None

    def read(self) -> float:
        value = self._from_proc()
        if value is None:
            value = self._from_iw()
        if value is None:
            raise RssiUnavailable(
                f"Interface {self.interface} reported no signal level. Is it "
                "connected to an access point? (RSSI is per-association.)"
            )
        return value


class MacRssi:
    """Reads RSSI via CoreWLAN (pyobjc), falling back to the legacy airport tool."""

    AIRPORT = (
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/"
        "Current/Resources/airport"
    )

    def __init__(self, interface: Optional[str] = None):
        self._iface = None
        try:
            import CoreWLAN  # type: ignore  # pip install pyobjc-framework-CoreWLAN

            client = CoreWLAN.CWWiFiClient.sharedWiFiClient()
            self._iface = client.interfaceWithName_(interface) if interface else client.interface()
        except ImportError:
            if not os.path.exists(self.AIRPORT):
                raise RssiUnavailable(
                    "On macOS 14.4+ the `airport` tool is gone. Install the CoreWLAN "
                    "bindings: pip install pyobjc-framework-CoreWLAN"
                )
        self.name = "macos:corewlan" if self._iface is not None else "macos:airport"

    def read(self) -> float:
        if self._iface is not None:
            value = self._iface.rssiValue()
            if not value:
                raise RssiUnavailable("CoreWLAN returned RSSI 0: WiFi off or not associated.")
            return float(value)
        out = subprocess.run([self.AIRPORT, "-I"], capture_output=True, text=True, timeout=2).stdout
        m = re.search(r"agrCtlRSSI:\s*(-?\d+)", out)
        if not m:
            raise RssiUnavailable("airport -I reported no RSSI: WiFi off or not associated.")
        return float(m.group(1))


class WindowsRssi:
    """Reads RSSI from `netsh wlan show interfaces`.

    Uses the `Rssi : -67` line (dBm) when Windows prints it; otherwise falls
    back to whole-percent quality converted as dBm ~= quality/2 - 100.
    """

    def __init__(self, interface: Optional[str] = None):
        self.interface = interface
        self.name = "windows:netsh"
        self.read()  # fail fast

    def read(self) -> float:
        out = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=3
        ).stdout
        blocks = re.split(r"\r?\n\s*\r?\n", out)
        for block in blocks:
            if self.interface and self.interface not in block:
                continue
            # Recent Windows 11 builds print the driver's dBm directly; prefer it.
            m = re.search(r"^\s*Rssi\s*:\s*(-\d+)", block, re.MULTILINE)
            if m:
                return float(m.group(1))
            m = re.search(r"^\s*Signal\s*:\s*(\d+)\s*%", block, re.MULTILINE)
            if m:
                return int(m.group(1)) / 2.0 - 100.0
        if "location" in out.lower():
            raise RssiUnavailable(
                "Windows is blocking WiFi info until Location is allowed. Open "
                "Settings > Privacy & security > Location, turn on Location services "
                "and 'Let desktop apps access your location', then retry."
            )
        raise RssiUnavailable("netsh reported no connected WiFi interface "
                              "(or a non-English Windows; this parser expects 'Signal').")


def open_source(interface: Optional[str]):
    system = platform.system()
    if system == "Linux":
        return LinuxRssi(interface)
    if system == "Darwin":
        return MacRssi(interface)
    if system == "Windows":
        return WindowsRssi(interface)
    raise RssiUnavailable(f"Unsupported OS: {system}")


# ---------------------------------------------------------------------------
# Detector (pure, no I/O)
# ---------------------------------------------------------------------------


class MotionDetector:
    """Rolling-variance detector with an empty-room calibrated threshold.

    threshold = max(mean + k*std, p99 * margin, floor) of the rolling variance
    observed during calibration. `hold` consecutive windows above threshold are
    required to raise MOTION, and the same number below to clear it.
    """

    def __init__(self, window: int, k: float = 4.0, margin: float = 1.5,
                 floor: float = 0.5, hold: int = 2):
        if window < 3:
            raise ValueError("window must cover at least 3 samples")
        self.window = window
        self.k, self.margin, self.floor, self.hold = k, margin, floor, hold
        self.samples: Deque[float] = collections.deque(maxlen=window)
        self.threshold: Optional[float] = None
        self.motion = False
        self._streak = 0

    def push(self, rssi: float) -> Optional[float]:
        """Add a sample; return the rolling variance once the window is full."""
        self.samples.append(rssi)
        if len(self.samples) < self.window:
            return None
        return statistics.pvariance(self.samples)

    def calibrate(self, baseline: List[float]) -> float:
        variances = []
        win: Deque[float] = collections.deque(maxlen=self.window)
        for x in baseline:
            win.append(x)
            if len(win) == self.window:
                variances.append(statistics.pvariance(win))
        if len(variances) < 5:
            raise ValueError("calibration too short for the chosen window")
        mean = statistics.fmean(variances)
        std = statistics.pstdev(variances)
        p99 = sorted(variances)[min(len(variances) - 1, int(0.99 * len(variances)))]
        self.threshold = max(mean + self.k * std, p99 * self.margin, self.floor)
        self.samples.clear()
        return self.threshold

    def update(self, variance: Optional[float]) -> bool:
        if variance is None or self.threshold is None:
            return self.motion
        above = variance > self.threshold
        if above != self.motion:
            self._streak += 1
            if self._streak >= self.hold:
                self.motion, self._streak = above, 0
        else:
            self._streak = 0
        return self.motion


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------


class Sampler(threading.Thread):
    def __init__(self, source, rate: float, detector: MotionDetector,
                 calibrate_s: float, csv_path: Optional[str], history: int):
        super().__init__(daemon=True)
        self.source, self.period, self.det = source, 1.0 / rate, detector
        self.calibrate_s, self.csv_path = calibrate_s, csv_path
        self.lock = threading.Lock()
        self.t: Deque[float] = collections.deque(maxlen=history)
        self.rssi: Deque[float] = collections.deque(maxlen=history)
        self.var: Deque[float] = collections.deque(maxlen=history)
        self.phase = "calibrating"
        self.error: Optional[str] = None
        self.stop_event = threading.Event()

    def run(self) -> None:
        writer = None
        fh = open(self.csv_path, "w", newline="") if self.csv_path else None
        if fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "rssi_dbm", "variance", "threshold", "motion", "phase"])
        try:
            self._loop(writer)
        except RssiUnavailable as exc:
            self.error = str(exc)
        finally:
            if fh:
                fh.close()
            self.stop_event.set()

    def _loop(self, writer) -> None:
        t0 = time.monotonic()
        baseline: List[float] = []
        changes, last = 0, None
        next_tick = t0
        print(f"[calibrate] Leave the room empty and still for {self.calibrate_s:.0f} s ...", flush=True)
        while not self.stop_event.is_set():
            now = time.monotonic()
            value = self.source.read()
            t = now - t0
            if last is not None and value != last:
                changes += 1
            last = value
            variance = motion = None
            if self.phase == "calibrating":
                baseline.append(value)
                if t >= self.calibrate_s:
                    thr = self.det.calibrate(baseline)
                    eff = changes / max(t, 1e-9)
                    print(f"[calibrate] {len(baseline)} samples, mean {statistics.fmean(baseline):.1f} dBm, "
                          f"threshold variance {thr:.3f} dB^2", flush=True)
                    print(f"[calibrate] driver RSSI changed {eff:.2f} times/s "
                          f"(requested {1 / self.period:.1f} samples/s)", flush=True)
                    if eff < 0.2:
                        print("[warning] RSSI barely changes on this driver; motion detection "
                              "will be slow or insensitive.", flush=True)
                    self.phase = "monitoring"
            else:
                variance = self.det.push(value)
                motion = self.det.update(variance)
                if variance is not None:
                    tag = "MOTION" if motion else "still "
                    print(f"{t:7.1f}s  {value:6.1f} dBm  var {variance:7.3f}  "
                          f"thr {self.det.threshold:.3f}  {tag}", flush=True)
            with self.lock:
                self.t.append(t)
                self.rssi.append(value)
                self.var.append(variance if variance is not None else math.nan)
            if writer:
                writer.writerow([f"{t:.3f}", value, "" if variance is None else f"{variance:.4f}",
                                 "" if self.det.threshold is None else f"{self.det.threshold:.4f}",
                                 "" if motion is None else int(motion), self.phase])
            next_tick += self.period
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))


def run_plot(sampler: Sampler) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.canvas.manager.set_window_title(f"RSSI motion monitor ({sampler.source.name})")
    (l_rssi,) = ax1.plot([], [], lw=1.2)
    (l_var,) = ax2.plot([], [], lw=1.2, label="rolling variance")
    l_thr = ax2.axhline(math.nan, color="tab:red", ls="--", label="threshold")
    ax1.set_ylabel("RSSI (dBm)")
    ax2.set_ylabel("variance (dB²)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left")

    def update(_):
        if sampler.stop_event.is_set():
            plt.close(fig)
            return ()
        with sampler.lock:
            t, r, v = list(sampler.t), list(sampler.rssi), list(sampler.var)
        if not t:
            return ()
        l_rssi.set_data(t, r)
        l_var.set_data(t, v)
        thr = sampler.det.threshold
        if thr is not None:
            l_thr.set_ydata([thr, thr])
        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()
        if sampler.phase == "calibrating":
            fig.suptitle("CALIBRATING: keep the room empty", color="tab:orange")
        elif sampler.det.motion:
            fig.suptitle("MOTION", color="tab:red", fontweight="bold")
        else:
            fig.suptitle("still", color="tab:green")
        return l_rssi, l_var, l_thr

    _anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--interface", help="WiFi interface (default: auto-detect)")
    p.add_argument("--rate", type=float, default=5.0, help="samples per second (default 5)")
    p.add_argument("--window", type=float, default=2.0, help="variance window in seconds (default 2)")
    p.add_argument("--calibrate", type=float, default=30.0, help="empty-room calibration seconds (default 30)")
    p.add_argument("--k", type=float, default=4.0, help="threshold = mean + k*std of baseline variance")
    p.add_argument("--floor", type=float, default=0.5, help="minimum variance threshold in dB^2")
    p.add_argument("--csv", help="also log samples to this CSV file")
    p.add_argument("--no-plot", action="store_true", help="console output only")
    args = p.parse_args(argv)
    if not (0.5 <= args.rate <= 50):
        p.error("--rate must be between 0.5 and 50")

    try:
        source = open_source(args.interface)
        first = source.read()
    except RssiUnavailable as exc:
        print(f"ERROR: no real WiFi RSSI available on {platform.system()}: {exc}", file=sys.stderr)
        print("Nothing was simulated. Run this on the laptop itself, connected to WiFi.", file=sys.stderr)
        return 2
    print(f"Source: {source.name}  first reading {first:.1f} dBm")

    window = max(3, round(args.window * args.rate))
    det = MotionDetector(window=window, k=args.k, floor=args.floor)
    history = int(args.rate * 120)
    sampler = Sampler(source, args.rate, det, args.calibrate, args.csv, history)
    sampler.start()

    plot = not args.no_plot
    if plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib not installed; console only (pip install matplotlib)")
            plot = False
    try:
        if plot:
            run_plot(sampler)
            sampler.stop_event.set()
        else:
            while not sampler.stop_event.wait(0.5):
                pass
    except KeyboardInterrupt:
        sampler.stop_event.set()
    sampler.join(timeout=2)
    if sampler.error:
        print(f"ERROR: {sampler.error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
