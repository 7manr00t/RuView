"""Tests for how provision.py protects the WiFi password and seed token (#1754).

The state file, the --dry-run NVS binary and the fallback CSV all hold the
secrets in clear, so they must be owner-only, and --state must not print them
unless asked. main() runs in-process with the NVS generator stubbed, so no
serial port is opened and no ESP-IDF tooling is needed. All credentials here
are placeholders.
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROVISION_PATH = Path(__file__).resolve().parents[1] / "provision.py"
SPEC = importlib.util.spec_from_file_location("provision", PROVISION_PATH)
provision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provision)

FAKE_PASSWORD = "fake-pass-not-real"
FAKE_TOKEN = "fake-token-not-real"
POSIX_ONLY = unittest.skipIf(sys.platform == "win32", "POSIX permission model only")


def mode_of(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="provision-secrets-")
        self.state_dir = os.path.join(self.root, "state")
        # The default umask is what made files 0644 in the first place.
        self.old_umask = os.umask(0o022)

    def tearDown(self):
        os.umask(self.old_umask)
        shutil.rmtree(self.root, ignore_errors=True)

    def run_main(self, *argv, cwd=None):
        """Run provision.main() with argv; return (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        code = 0
        old_cwd = os.getcwd()
        try:
            os.chdir(cwd or self.root)
            with mock.patch.object(sys, "argv", ["provision.py", *argv]), \
                    contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    provision.main()
                except SystemExit as exc:
                    code = exc.code
        finally:
            os.chdir(old_cwd)
        return code, out.getvalue(), err.getvalue()


@POSIX_ONLY
class TestStateFilePermissions(_TempDirCase):
    def test_new_state_dir_and_file_are_owner_only(self):
        path = provision.save_state("COM7", self.state_dir, {"password": FAKE_PASSWORD})
        self.assertEqual(mode_of(self.state_dir), 0o700)
        self.assertEqual(mode_of(path), 0o600)

    def test_save_tightens_files_left_by_an_older_version(self):
        os.makedirs(self.state_dir)
        os.chmod(self.state_dir, 0o755)
        legacy = provision._state_path_for("/dev/ttyUSB0", self.state_dir)
        stale_tmp = legacy + ".tmp"
        for path in (legacy, stale_tmp):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"password": FAKE_PASSWORD}, f)
            os.chmod(path, 0o644)

        provision.save_state("COM7", self.state_dir, {"ssid": "test-ssid"})

        self.assertEqual(mode_of(self.state_dir), 0o700)
        self.assertEqual(mode_of(legacy), 0o600)
        self.assertEqual(mode_of(stale_tmp), 0o600)

    def test_planted_symlink_at_old_temp_path_does_not_receive_secret(self):
        os.makedirs(self.state_dir, mode=0o700)
        victim = os.path.join(self.root, "victim.txt")
        with open(victim, "w", encoding="utf-8") as f:
            f.write("original")
        os.symlink(victim, provision._state_path_for("COM7", self.state_dir) + ".tmp")

        provision.save_state("COM7", self.state_dir, {"password": FAKE_PASSWORD})

        with open(victim, encoding="utf-8") as f:
            self.assertEqual(f.read(), "original")

    def test_hardening_does_not_chmod_through_a_symlink(self):
        os.makedirs(self.state_dir, mode=0o700)
        victim = os.path.join(self.root, "victim.txt")
        with open(victim, "w", encoding="utf-8") as f:
            f.write("not a state file")
        os.chmod(victim, 0o644)
        os.symlink(victim, os.path.join(self.state_dir, "link.json"))

        with contextlib.redirect_stderr(io.StringIO()):
            provision.harden_state_dir(self.state_dir)

        self.assertEqual(mode_of(victim), 0o644)

    def test_state_inspection_repairs_legacy_permissions(self):
        # --state never writes, so the repair has to happen before the read.
        os.makedirs(self.state_dir)
        os.chmod(self.state_dir, 0o755)
        legacy = provision._state_path_for("COM7", self.state_dir)
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump({"password": FAKE_PASSWORD}, f)
        os.chmod(legacy, 0o644)

        code, _, _ = self.run_main("--port", "COM7", "--state-dir", self.state_dir, "--state")

        self.assertEqual(code, 0)
        self.assertEqual(mode_of(self.state_dir), 0o700)
        self.assertEqual(mode_of(legacy), 0o600)


class TestStateOutputMasking(_TempDirCase):
    def setUp(self):
        super().setUp()
        provision.save_state("COM7", self.state_dir, {
            "ssid": "test-ssid",
            "password": FAKE_PASSWORD,
            "seed_token": FAKE_TOKEN,
            "target_ip": "192.0.2.10",
        })

    def test_state_hides_password_and_seed_token(self):
        code, out, err = self.run_main(
            "--port", "COM7", "--state-dir", self.state_dir, "--state")

        self.assertEqual(code, 0)
        self.assertNotIn(FAKE_PASSWORD, out)
        self.assertNotIn(FAKE_TOKEN, out)
        shown = json.loads(out)
        self.assertEqual(shown["password"], "(set)")
        self.assertEqual(shown["seed_token"], "(set)")
        self.assertEqual(shown["ssid"], "test-ssid")
        self.assertIn("--show-secrets", err)

    def test_show_secrets_prints_them(self):
        code, out, _ = self.run_main(
            "--port", "COM7", "--state-dir", self.state_dir, "--state", "--show-secrets")

        self.assertEqual(code, 0)
        shown = json.loads(out)
        self.assertEqual(shown["password"], FAKE_PASSWORD)
        self.assertEqual(shown["seed_token"], FAKE_TOKEN)

    def test_redact_marks_empty_and_leaves_absent_alone(self):
        shown = provision.redact_secrets({"password": "", "ssid": "test-ssid"})
        self.assertEqual(shown, {"password": "(empty)", "ssid": "test-ssid"})


@POSIX_ONLY
class TestIntermediateArtifacts(_TempDirCase):
    ARGS = ("--port", "COM7", "--ssid", "test-ssid", "--password", FAKE_PASSWORD,
            "--target-ip", "192.0.2.10", "--dry-run")

    def test_dry_run_binary_is_owner_only_even_if_it_already_existed(self):
        out_path = os.path.join(self.root, "nvs_provision.bin")
        with open(out_path, "wb") as f:
            f.write(b"old")
        os.chmod(out_path, 0o644)

        with mock.patch.object(provision, "generate_nvs_binary",
                               return_value=b"nvs-with-" + FAKE_PASSWORD.encode()):
            code, _, _ = self.run_main(*self.ARGS, "--state-dir", self.state_dir)

        self.assertIn(code, (0, None))
        self.assertEqual(mode_of(out_path), 0o600)

    def test_fallback_csv_is_owner_only(self):
        with mock.patch.object(provision, "generate_nvs_binary",
                               side_effect=RuntimeError("generator missing")):
            code, _, _ = self.run_main(*self.ARGS, "--state-dir", self.state_dir)

        self.assertEqual(code, 1)
        csv_path = os.path.join(self.root, "nvs_config.csv")
        self.assertEqual(mode_of(csv_path), 0o600)

    def test_generator_files_live_in_a_private_dir(self):
        seen = {}

        def fake_generator(cmd, **_kwargs):
            csv_path, bin_path = cmd[-3], cmd[-2]
            seen["dir_mode"] = mode_of(os.path.dirname(bin_path))
            seen["csv_mode"] = mode_of(csv_path)
            with open(bin_path, "wb") as f:
                f.write(b"nvs")

        # A shared temp dir like Linux /tmp, rather than macOS's per-user one.
        shared_tmp = os.path.join(self.root, "shared-tmp")
        os.makedirs(shared_tmp)
        os.chmod(shared_tmp, 0o755)
        with mock.patch.object(provision.tempfile, "tempdir", shared_tmp), \
                mock.patch.object(provision.subprocess, "check_call",
                                  side_effect=fake_generator):
            self.assertEqual(provision.generate_nvs_binary("key,type\n", 0x6000), b"nvs")

        self.assertEqual(seen["dir_mode"], 0o700)
        self.assertEqual(seen["csv_mode"], 0o600)


if __name__ == "__main__":
    unittest.main()
