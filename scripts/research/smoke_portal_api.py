#!/usr/bin/env python3
"""Exercise the local portal and its initially open GESTUR access point.

Uses only the configured AP returned by /api/wifi; never scans other networks.
Temporary Wi-Fi passwords are random and remain in memory; restoration of the
original open state is attempted and verified in finally. Use Ethernet because
Wi-Fi is restarted.
No credentials or literal SSIDs are written to the result.
"""

import argparse
import http.cookiejar
import json
import os
import re
import secrets
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1"
TOKEN_PATH = Path("/etc/gestur/portal-token")
SSID_PATTERN = re.compile(r"GESTUR-[A-F0-9]{4}")
READINESS_TIMEOUT = 15
JOB_TIMEOUT = 90
POLL_INTERVAL = 1


class SmokeFailure(Exception):
    """Only fixed, credential-free failure codes belong in this exception."""


class API:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(self.cookies),
        )

    def request(self, method, endpoint, body=None, expected=200, timeout=10):
        headers = {"Accept": "application/json", "X-Gestur-Request": "1"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + "/api/" + endpoint,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                if response.status != expected:
                    raise SmokeFailure("unexpected_http_status")
                raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise SmokeFailure("response_too_large")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise SmokeFailure("invalid_json_object")
                return result
        except urllib.error.HTTPError as error:
            # Never print response bodies, request headers, URLs or exceptions.
            raise SmokeFailure("http_" + str(error.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SmokeFailure("connection_failed") from None
        except (ValueError, UnicodeError):
            raise SmokeFailure("invalid_json") from None


def require(condition, code):
    if not condition:
        raise SmokeFailure(code)


def wait_ready(api, timeout=READINESS_TIMEOUT):
    """Only startup connection failures are retried; later API errors propagate."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            state = api.request("GET", "session", timeout=min(2, max(0.01, remaining)))
        except SmokeFailure as error:
            if str(error) != "connection_failed":
                raise
        else:
            require(
                type(state.get("authenticated")) is bool, "invalid_readiness_response"
            )
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.25, remaining))
    raise SmokeFailure("portal_readiness_timeout_15s")


def wifi_is(state, ssid, secured):
    return (
        state.get("ssid") == ssid
        and state.get("secured") is secured
        and state.get("active") is True
    )


def wait_job(api, ssid, secured, *, allow_failed=False, timeout=JOB_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            state = api.request("GET", "wifi", timeout=min(10, max(0.1, remaining)))
        except SmokeFailure as error:
            if str(error) != "connection_failed":
                raise
        else:
            job = state.get("job") or {}
            status = job.get("state")
            if status not in ("pending", "applying"):
                if allow_failed:
                    return state
                require(status == "completed", "wifi_job_not_completed")
                require(job.get("ssid") == ssid, "wifi_job_ssid_changed")
                require(wifi_is(state, ssid, secured), "wifi_state_mismatch")
                return state
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(POLL_INTERVAL, remaining))
    raise SmokeFailure("wifi_job_deadline_90s")


def apply_wifi(api, ssid, password):
    started = time.monotonic()
    result = api.request(
        "PUT", "wifi", {"ssid": ssid, "password": password}, expected=202
    )
    job = result.get("job") or {}
    require(
        job.get("state") == "pending" and job.get("ssid") == ssid,
        "wifi_job_not_accepted",
    )
    wait_job(
        api,
        ssid,
        password is not None,
        timeout=max(0.1, JOB_TIMEOUT - (time.monotonic() - started)),
    )


def run(api, token):
    started = time.monotonic()
    report = {"ok": False, "checks": [], "wifi_restored_open": None}
    authenticated = False
    mutation_attempted = False
    original_ssid = None
    baseline_config = None

    def check(name, **details):
        if "ssid" in details:
            details.pop("ssid")
            details["ssid_matches_original"] = True
        report["checks"].append({"check": name, "ok": True, **details})

    try:
        wait_ready(api)
        session = api.request("POST", "session", {"token": token})
        require(session.get("authenticated") is True, "login_failed")
        authenticated = True
        token = None
        require(
            api.request("GET", "session").get("authenticated") is True,
            "session_cookie_not_retained",
        )
        check("login_cookie_session")

        models = api.request("GET", "models")
        require(
            models.get("models") == [] and models.get("active") is None,
            "expected_empty_model_library",
        )
        check("models_empty", count=0, active=None)

        runtime = api.request("GET", "runtime")
        require(runtime.get("online") is True, "runtime_offline")
        require(
            runtime.get("selected_model") is None
            and runtime.get("rendered_model") is None,
            "unexpected_runtime_model",
        )
        require(runtime.get("error") is None, "runtime_reports_error")
        check("runtime_online", online=True, render_fps=runtime.get("render_fps"))

        baseline_config = api.request("GET", "config")
        current, defaults = (
            baseline_config.get("config"),
            baseline_config.get("defaults"),
        )
        for item in (current, defaults):
            require(
                isinstance(item, dict) and item.get("schema_version") == 1,
                "invalid_configuration_response",
            )
            require(item.get("active_model") is None, "unexpected_config_model")
            require(
                all(
                    isinstance(item.get(key), dict)
                    for key in ("tracking", "render", "controls")
                ),
                "missing_configuration_sections",
            )
        check("configuration_read", current_equals_defaults=current == defaults)

        initial = api.request("GET", "wifi")
        ssid = initial.get("ssid")
        require(
            isinstance(ssid, str) and SSID_PATTERN.fullmatch(ssid),
            "initial_access_point_not_default_gestur_ssid",
        )
        require(
            wifi_is(initial, ssid, False),
            "initial_access_point_not_expected_open_active",
        )
        require(
            (initial.get("job") or {}).get("state") not in ("pending", "applying"),
            "preexisting_wifi_job",
        )
        original_ssid = initial["ssid"]
        check("wifi_initial", ssid=original_ssid, secured=False, active=True)

        # Set before PUT: a lost response does not prove the mutation was rejected.
        mutation_attempted = True
        for name in ("wifi_add_password", "wifi_change_password"):
            apply_wifi(api, original_ssid, secrets.token_urlsafe(24))
            check(name, ssid=original_ssid, secured=True, active=True)
        apply_wifi(api, original_ssid, None)
        check("wifi_remove_password", ssid=original_ssid, secured=False, active=True)
        report["ok"] = True
    except SmokeFailure as error:
        report["failure"] = str(error)
    except KeyboardInterrupt:
        report["failure"] = "interrupted"
    except Exception:
        report["failure"] = "unexpected_error"
    finally:
        # A failed/timed-out test may still have a running server job. Let it
        # settle before queuing restoration; never race it with a direct helper.
        if mutation_attempted:
            try:
                state = wait_job(api, original_ssid, False, allow_failed=True)
                if not wifi_is(state, original_ssid, False):
                    apply_wifi(api, original_ssid, None)
                state = api.request("GET", "wifi")
                require(wifi_is(state, original_ssid, False), "restore_state_mismatch")
                report["wifi_restored_open"] = True
                check("wifi_final", ssid=original_ssid, secured=False, active=True)
            except BaseException:
                report["ok"] = False
                report["wifi_restored_open"] = False
                report["restore_failure"] = "could_not_verify_open_active_original_ssid"
        if baseline_config is not None:
            try:
                require(
                    api.request("GET", "config") == baseline_config,
                    "configuration_changed",
                )
                check("configuration_unchanged")
            except BaseException:
                report["ok"] = False
                report["config_check_failure"] = (
                    "could_not_verify_configuration_unchanged"
                )
        if authenticated:
            try:
                require(
                    api.request("DELETE", "session").get("authenticated") is False,
                    "logout_failed",
                )
                check("session_logout")
            except BaseException:
                report["ok"] = False
                report["logout_failure"] = "could_not_verify_logout"
        api.cookies.clear()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=BASE_URL, help="Local HTTP portal; default port 80"
    )
    parser.add_argument("--token-file", type=Path, default=TOKEN_PATH)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.base_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost", "::1")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        parser.error("--base-url must be local HTTP without credentials or a path")

    # Convert a normal termination into a finally path, including Wi-Fi cleanup.
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    if os.geteuid() != 0:
        report = {"ok": False, "failure": "run_locally_as_root"}
    else:
        try:
            token = args.token_file.read_text().strip()
            require(len(token) >= 24, "invalid_admin_token_file")
            report = run(API(args.base_url), token)
        except BaseException:
            report = {"ok": False, "failure": "unable_to_start_smoke_test"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
