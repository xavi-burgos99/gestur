"""Exercise installer arguments without root, package downloads or host changes."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def invoke_main(*args):
    return subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
shift
install_gestur() {
    printf 'INSTALL %s %s\\n' "$DEBIAN_FRONTEND" "$PIP_NO_INPUT"
    printf '<%s>\\n' "$@"
    if read -r ignored; then exit 90; fi
}
uninstall_gestur() { printf 'UNINSTALL\\n'; }
main "$@"
""",
            "_",
            str(ROOT / "gestur.sh"),
            *args,
        ],
        input="No installer should consume this input\n",
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("hostname", ["a", "sala-2", "7", "a" * 63])
def test_hostname_is_forwarded_literally_and_installation_has_no_interactive_input(
    hostname,
):
    result = invoke_main("install", "--hostname", hostname)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"INSTALL noninteractive 1\n<--hostname>\n<{hostname}>\n"


def test_default_install_does_not_override_hostname_and_uninstall_remains_available():
    result = invoke_main("install")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "INSTALL noninteractive 1\n<>\n"
    result = invoke_main("uninstall")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "UNINSTALL\n"


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("invalid",),
        ("install", "--unknown"),
        ("install", "--hostname"),
        ("install", "--hostname", "sala", "--hostname", "otra"),
        ("uninstall", "--hostname", "sala"),
        *(
            ("install", "--hostname", value)
            for value in (
                "",
                "Sala",
                "sala.local",
                "-sala",
                "sala-",
                "a" * 64,
                "a b",
                "a/b",
                "sala;id",
                "sala\n",
            )
        ),
    ],
)
def test_invalid_arguments_are_rejected_before_any_install_or_uninstall(args):
    result = invoke_main(*args)
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], "/opt/gestur|"),
        (["/opt/gestur"], "/opt/gestur|"),
        (["/opt/gestur", "--hostname", "sala-2"], "/opt/gestur|sala-2"),
        (["--hostname", "sala-2"], "/opt/gestur|sala-2"),
    ],
)
def test_portal_hook_parses_the_same_optional_hostname_before_root_commands(
    args, expected
):
    # Run the real argument parser only, before its first privileged check.
    prefix = (
        (ROOT / "scripts/install-portal.sh").read_text().split("if [[ $(id -u)", 1)[0]
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            prefix + '\nprintf "%s|%s" "$INSTALL_ROOT" "$TASK_HOSTNAME"',
            "_",
            *args,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_portal_sudo_grants_exact_device_actions_and_never_bootstrap():
    script = (ROOT / "scripts/install-portal.sh").read_text()
    rules = [
        line
        for line in script.splitlines()
        if line.startswith("printf '%s\\n' ") and "gestur-device " in line
    ]
    assert len(rules) == 1
    actions = rules[0].split("NOPASSWD: ", 1)[1].split("' >", 1)[0].split(", ")
    assert actions == [
        f"/usr/local/libexec/gestur-device {action}"
        for action in ("status", "hostname", "onboarding", "reset")
    ]
    assert "cat /etc/gestur/portal-token" not in script
    assert "secrets.token_urlsafe" not in script
    assert (
        'install -o root -g root -m 644 "$INSTALL_ROOT/config/default.json" /etc/gestur/default.json'
        in script
    )
    assert (
        'install -o root -g root -m 755 "$INSTALL_ROOT/scripts/gestur-device.py" /usr/local/libexec/gestur-device'
        in script
    )
