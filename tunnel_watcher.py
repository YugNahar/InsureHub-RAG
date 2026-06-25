#!/usr/bin/env python3
"""
tunnel_watcher.py — keeps cloudflared alive and auto-updates Vercel when the URL changes.

Run once on startup:
    python3 tunnel_watcher.py

Run in background (logs to ~/.cloudflare_tunnel.log):
    nohup python3 tunnel_watcher.py &
    # or
    python3 tunnel_watcher.py --daemon
"""

import os
import re
import subprocess
import sys
import time
import logging
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────────
BACKEND_PORT       = 8501
CLOUDFLARED_CMD    = ["cloudflared", "tunnel", "--url", f"http://localhost:{BACKEND_PORT}"]
METRICS_URL        = "http://localhost:20241/metrics"
# Written into the Docker-mounted app/ dir so the backend container can read it
TUNNEL_URL_FILE    = os.path.join(os.path.dirname(__file__), "RAG_InsureAI", "app", "tunnel_url.txt")
LOG_FILE           = os.path.expanduser("~/.cloudflare_tunnel.log")

# Main Layla chat frontend (Vite/React project)
VERCEL_PROJECT_DIR = os.path.expanduser(
    "~/Downloads/insurehub-RAG-frontend/insurehub-your-ai-insurance-advisor"
)
# Admin + Agent panels (plain HTML, deployed separately to Vercel)
PANELS_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels")

POLL_INTERVAL      = 10   # seconds between URL checks
STARTUP_WAIT       = 20   # seconds to wait after starting cloudflared before polling

_ENV = {**os.environ, "PATH": "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [tunnel-watcher] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_tunnel_url() -> str | None:
    """Read current tunnel hostname from cloudflared's local metrics endpoint."""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=3) as r:
            text = r.read().decode()
        m = re.search(r'userHostname="(https://[^"]+trycloudflare\.com)"', text)
        return m.group(1) if m else None
    except Exception:
        return None


def cf_is_running(proc) -> bool:
    return proc is not None and proc.poll() is None


def start_cloudflared():
    """Kill any stray cloudflared processes, then start a fresh one."""
    subprocess.run(["pkill", "-f", "cloudflared tunnel --url"], capture_output=True)
    time.sleep(1)
    proc = subprocess.Popen(
        CLOUDFLARED_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info("cloudflared started (pid=%d) — waiting %ds for URL…", proc.pid, STARTUP_WAIT)
    time.sleep(STARTUP_WAIT)
    return proc


def _vercel_set_env(project_dir: str, url: str, vars: tuple):
    """Set env vars and trigger a prod redeploy for a single Vercel project."""
    if not os.path.isdir(project_dir):
        log.warning("  Skipping %s — directory not found", project_dir)
        return
    label = os.path.basename(project_dir)
    for var in vars:
        result = subprocess.run(
            ["vercel", "env", "add", var, "production", "--force"],
            input=url.encode(),
            cwd=project_dir,
            env=_ENV,
            capture_output=True,
        )
        if result.returncode == 0:
            log.info("  ✓ [%s] %s updated", label, var)
        else:
            log.warning("  ✗ [%s] %s failed: %s", label, var, result.stderr.decode().strip())

    subprocess.Popen(
        ["vercel", "--prod"],
        cwd=project_dir,
        env=_ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info("  Redeploy queued for [%s].", label)


def update_vercel(url: str):
    """Push new tunnel URL to all Vercel projects and trigger redeployments."""
    log.info("Updating Vercel deployments → %s", url)

    # 1. Main Layla chat frontend (uses both VITE_API_BASE_URL and VITE_API_URL)
    _vercel_set_env(VERCEL_PROJECT_DIR, url, ("VITE_API_BASE_URL", "VITE_API_URL"))

    # 2. Admin + Agent panels (only VITE_API_BASE_URL — build.js reads this)
    _vercel_set_env(PANELS_PROJECT_DIR, url, ("VITE_API_BASE_URL",))


def write_url_file(url: str):
    for path in (TUNNEL_URL_FILE, os.path.expanduser("~/.cloudflare_tunnel_url")):
        try:
            with open(path, "w") as f:
                f.write(url)
        except OSError:
            pass


def _read_panels_vercel_url() -> str:
    """Try to read the panels Vercel deployment URL from .vercel/project.json."""
    try:
        import json
        proj_file = os.path.join(PANELS_PROJECT_DIR, ".vercel", "project.json")
        with open(proj_file) as f:
            data = json.load(f)
        # Vercel project.json contains {"projectId":..., "orgId":..., "settings":{...}}
        # The live URL is not stored here, so we construct a best-guess from the project name
        name = data.get("projectName") or data.get("name", "")
        if name:
            return f"https://{name}.vercel.app"
    except Exception:
        pass
    return "(deploy panels/ to Vercel first — see README)"


def print_links(url: str):
    panels_url = _read_panels_vercel_url()
    line = "─" * 66
    print(f"\n{line}")
    print(f"  Tunnel (backend) : {url}")
    print(f"  User chat        : https://insurehub-your-ai-insurance-advisor.vercel.app")
    print(f"  Admin panel      : {panels_url}/admin")
    print(f"  Agent dashboard  : {panels_url}/agent-dashboard")
    print(f"{line}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    current_url = None
    proc = None

    log.info("Tunnel watcher started.")

    # If cloudflared is already running externally, read its URL first
    existing_url = get_tunnel_url()
    if existing_url:
        log.info("Detected existing tunnel: %s", existing_url)
        current_url = existing_url
        write_url_file(current_url)
        print_links(current_url)
        # Find the existing process so we can watch it
        result = subprocess.run(
            ["pgrep", "-f", "cloudflared tunnel --url"], capture_output=True, text=True
        )
        if result.stdout.strip():
            class _FakeProc:
                def __init__(self, pid): self._pid = pid
                def poll(self):
                    try:
                        os.kill(self._pid, 0)
                        return None   # still alive
                    except ProcessLookupError:
                        return 1      # dead
            try:
                proc = _FakeProc(int(result.stdout.strip().splitlines()[0]))
            except ValueError:
                proc = None
    else:
        proc = start_cloudflared()

    while True:
        # Restart cloudflared if it died
        if not cf_is_running(proc):
            log.warning("cloudflared is down — restarting…")
            proc = start_cloudflared()

        url = get_tunnel_url()
        if url and url != current_url:
            log.info("URL changed: %s → %s", current_url or "(none)", url)
            current_url = url
            write_url_file(current_url)
            print_links(current_url)
            update_vercel(current_url)

        time.sleep(POLL_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--daemon" in sys.argv:
        # Redirect all output to log file and fork to background
        pid = os.fork()
        if pid > 0:
            print(f"Tunnel watcher running in background (PID {pid})")
            print(f"Logs: {LOG_FILE}")
            sys.exit(0)
        # Child: redirect stdio to log file
        sys.stdout.flush()
        sys.stderr.flush()
        log_fd = open(LOG_FILE, "a")
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [tunnel-watcher] %(message)s",
            datefmt="%H:%M:%S",
            force=True,
            handlers=[logging.StreamHandler(log_fd)],
        )
    run()
