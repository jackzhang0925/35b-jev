"""TaskEnv: one Harbor task as an RL environment, backed by a Modal sandbox.

Why this file exists
--------------------
`bridge.HarborTask` used to shell out to `docker run` / `docker exec`. Docker is not
installed on every rollout host (this one has none), so the same four-method interface
is now served by a `modal.Sandbox` instead:

    env = make_env(task_dir)     # RL_ENV=modal (default) | docker
    obs = env.reset()            # -> the task's instruction.md text
    obs = env.exec("ls /app")    # -> stdout+stderr, tail-truncated
    reward = env.grade()         # -> 1.0 / 0.0, the task's own verifier
    env.close()

Two implementations of the same `TaskEnv` protocol:

    ModalTask          one `modal.Sandbox` per episode (the default; no Docker needed)
    LocalDockerTask    the original docker run/exec behaviour, kept for Linux hosts

Public Modal APIs only: `modal.Image.from_registry / from_dockerfile`, `Image.build`,
`modal.App.lookup`, `modal.Sandbox.create`, `sandbox.exec`, `sandbox.terminate`.

Image choice for a task
-----------------------
1. `[environment].docker_image` in `task.toml` -> `modal.Image.from_registry(tag)`
   (all of `data/tb2/terminal-bench-2/*`).
2. otherwise the task's own `environment/Dockerfile` -> `modal.Image.from_dockerfile`
   with `context_dir=environment/` (all of `data/train/aider-polyglot/*`, whose
   `task.toml` has no `docker_image`). Modal builds it remotely; no local Docker.
3. otherwise `debian:bookworm-slim`, so the interface still works for a stub task.

The task's `tests/` directory is mounted at `/tests` (Harbor's own convention), so the
verifier runs exactly as it does under Harbor.

Grading
-------
`grade()` runs the task's verifier -- `tests/test.sh`, else `tests/run-tests.sh`, else a
root-level `test.sh` / `run-tests.sh` -- from `/app`, and returns 1.0 iff it passed.

Which verdict is used matters: every verifier in this dataset ends with `exit 0` and
records its answer in `/logs/verifier/reward.txt` (TB2's `tests/test.sh` ends with
`if [ $? -eq 0 ]; then echo 1 > .../reward.txt; else echo 0 > ...; fi`, and the
aider-polyglot one always exits 0). So the exit code alone would score every rollout
1.0 -- note that `bridge.py`'s old `bash /tests/run-tests.sh | tail -50` check was
masking the verifier's status *and* its exit code that way. `verdict()` therefore reads
`reward.txt` when the verifier wrote one (this is exactly the number Harbor puts in
`result.json.verifier_result.rewards.reward`, see eval/run_eval.py) and falls back to
`exit code == 0` when it did not.

Smoke test (spends a few cents of Modal credits):

    python rl/envs.py --smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# -- limits -----------------------------------------------------------------
TAIL_CHARS = 8000          # observation window, same number bridge.py used
KEEP_TAIL = 50             # verifier lines kept for debugging (bridge.py's `tail -50`)
EXEC_TIMEOUT = 600         # seconds, per shell command
GRADE_TIMEOUT = 900        # seconds, per verifier run
SANDBOX_TIMEOUT = 3600     # seconds, whole episode; must outlive MAX_TURNS execs
IDLE_TIMEOUT = 900         # seconds, so a crashed rollout cannot bill forever
APP_NAME = "rl-task-envs"
TESTS_MOUNT = "/tests"
TESTS_DIR = "tests"
REWARD_PATH = "/logs/verifier/reward.txt"
REWARD_DIR = "/logs/verifier"     # POSIX string: this path lives in the container, not on Windows
FALLBACK_IMAGE = "debian:bookworm-slim"

# Verifier lookup order; the first that exists in the task directory wins.
TEST_SCRIPTS = (
    "tests/test.sh",
    "tests/run-tests.sh",
    "test.sh",
    "run-tests.sh",
)


# ---------------------------------------------------------------------------
# Task spec: what task.toml, Dockerfile and tests/ say about one task
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """Everything an env needs to boot one task, read from the task directory."""

    task_dir: Path
    name: str
    docker_image: str | None = None
    dockerfile: Path | None = None
    test_script: str | None = None
    cfg: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, task_dir: str | Path) -> TaskSpec:
        task_dir = Path(task_dir)
        cfg: dict[str, Any] = {}
        cfg_path = task_dir / "task.toml"
        if cfg_path.is_file():
            with cfg_path.open("rb") as f:
                cfg = tomllib.load(f)
        environment = cfg.get("environment") or {}
        dockerfile = task_dir / "environment" / "Dockerfile"
        script = next((rel for rel in TEST_SCRIPTS if (task_dir / rel).is_file()), None)
        return cls(
            task_dir=task_dir,
            name=(cfg.get("task") or {}).get("name") or task_dir.name,
            docker_image=environment.get("docker_image"),
            dockerfile=dockerfile if dockerfile.is_file() else None,
            test_script=script,
            cfg=cfg,
        )

    # -- derived facts -----------------------------------------------------
    @property
    def instruction(self) -> str:
        return (self.task_dir / "instruction.md").read_text(encoding="utf8")

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / TESTS_DIR

    @property
    def image_label(self) -> str:
        if self.docker_image:
            return self.docker_image
        if self.dockerfile:
            return f"dockerfile:{self.dockerfile}"
        return FALLBACK_IMAGE

    def sandbox_timeout(self) -> int:
        """Episode budget: never shorter than SANDBOX_TIMEOUT, and outlives agent.timeout_sec."""
        agent = self.cfg.get("agent") or {}
        try:
            agent_secs = int(float(agent.get("timeout_sec") or 0))
        except (TypeError, ValueError):
            agent_secs = 0
        return max(SANDBOX_TIMEOUT, agent_secs + 300)

    def resources(self) -> dict[str, Any]:
        """cpu/memory hints for Sandbox.create, straight from `[environment]`."""
        environment = self.cfg.get("environment") or {}
        out: dict[str, Any] = {}
        cpus = environment.get("cpus")
        if isinstance(cpus, (int, float)) and not isinstance(cpus, bool) and cpus > 0:
            out["cpu"] = float(cpus)
        memory_mb = environment.get("memory_mb")
        if isinstance(memory_mb, int) and not isinstance(memory_mb, bool) and memory_mb > 0:
            out["memory"] = memory_mb
        return out


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
@runtime_checkable
class TaskEnv(Protocol):
    """What `bridge.HarborTask` and `loop.py` need from one task container."""

    def reset(self) -> str:
        """Boot the task's container and return the instruction text."""
        ...

    def exec(self, command: str) -> str:
        """Run `command` in the container; return stdout+stderr, tail-truncated."""
        ...

    def grade(self) -> float:
        """Run the task's verifier; 1.0 iff it passed, else 0.0."""
        ...

    def close(self) -> None:
        """Tear the container down. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    return (text or "")[-limit:]


def resolve_task_dir(task_dir: str | Path) -> Path:
    """Accept a path relative to the cwd or to the repo root (rl/../)."""
    path = Path(task_dir)
    if path.is_dir():
        return path
    root = Path(__file__).resolve().parent.parent
    candidate = root / path
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"task directory not found: {task_dir}")


def grade_command(spec: TaskSpec) -> str | None:
    """Shell snippet that runs the task's verifier and prints a parseable verdict.

    The verifier's exit status is captured through PIPESTATUS[0] because its output is
    piped through `tail`; `reward.txt` is removed first so a stale value from an earlier
    run cannot decide this one.
    """
    if spec.test_script is None:
        return None
    script = f"{TESTS_MOUNT}/{Path(spec.test_script).name}"
    return (
        f"mkdir -p {REWARD_DIR} && rm -f {REWARD_PATH} && cd /app 2>/dev/null; "
        f"TEST_DIR={TESTS_MOUNT} bash {script} 2>&1 | tail -{KEEP_TAIL}; "
        'ec=${PIPESTATUS[0]}; '
        'echo "__RL_EXIT__=${ec}"; '
        f'echo "__RL_REWARD__=$(cat {REWARD_PATH} 2>/dev/null)"'
    )


def _marker(text: str, tag: str) -> str | None:
    for line in reversed((text or "").splitlines()):
        if line.startswith(tag):
            return line[len(tag):].strip()
    return None


def verdict(output: str) -> float:
    """1.0 iff the verifier passed: its reward.txt when written, else exit code 0."""
    reward = _marker(output, "__RL_REWARD__=")
    if reward:
        try:
            return 1.0 if float(reward) >= 1.0 else 0.0
        except ValueError:
            pass
    exit_code = _marker(output, "__RL_EXIT__=")
    try:
        return 1.0 if int(exit_code) == 0 else 0.0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _drain(proc) -> tuple[str, str, int]:
    """Read a Modal `ContainerProcess` to completion -> (stdout, stderr, exit code)."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        out = pool.submit(proc.stdout.read)
        err = pool.submit(proc.stderr.read)
        code = proc.wait()
        return (out.result() or ""), (err.result() or ""), int(code or 0)


# ---------------------------------------------------------------------------
# Modal-backed env (default)
# ---------------------------------------------------------------------------
_APPS: dict[str, Any] = {}  # one modal.App per process: App.lookup is a round trip


def _modal_app(modal, name: str = APP_NAME):
    if name not in _APPS:
        _APPS[name] = modal.App.lookup(name, create_if_missing=True)
    return _APPS[name]


class ModalTask:
    """One task container as a Modal sandbox.

    API calls (modal 1.6.4):
        app     = modal.App.lookup(APP_NAME, create_if_missing=True)
        image   = modal.Image.from_registry(tag) | modal.Image.from_dockerfile(...)
        image   = image.add_local_dir(tests, "/tests")     # mounted, no build step
        image.build(app)                                   # eager, so boot time is measured alone
        sandbox = modal.Sandbox.create("sleep", "infinity", image=image, app=app, timeout=...)
        proc    = sandbox.exec("bash", "-lc", cmd, timeout=...)
        sandbox.terminate()

    `boot_seconds` / `build_seconds` are recorded so `--smoke` can report the cost of
    image-backed sandbox creation separately from the one-off image build.
    """

    def __init__(
        self,
        task_dir: str | Path,
        *,
        force_image: str | None = None,
        timeout: int | None = None,
        exec_timeout: int = EXEC_TIMEOUT,
        grade_timeout: int = GRADE_TIMEOUT,
    ) -> None:
        self.spec = TaskSpec.load(resolve_task_dir(task_dir))
        self.task_dir = self.spec.task_dir
        self.name = self.spec.name
        self.timeout = timeout or self.spec.sandbox_timeout()
        self.exec_timeout = exec_timeout
        self.grade_timeout = grade_timeout
        self.force_image = force_image
        self.sandbox: Any | None = None
        self.origin: str | None = None
        self.boot_seconds: float | None = None
        self.build_seconds: float | None = None
        self._app: Any | None = None
        self._image: Any | None = None

    # -- plumbing ----------------------------------------------------------
    @staticmethod
    def _modal():
        import modal  # lazy: `rl/loop.py --dry-run` must not need Modal installed

        return modal

    @property
    def app(self):
        if self._app is None:
            self._app = _modal_app(self._modal())
        return self._app

    def image(self):
        """The task image (registry tag, else its Dockerfile), with /tests added."""
        if self._image is not None:
            return self._image
        modal = self._modal()
        if self.force_image:
            image = modal.Image.from_registry(self.force_image)
            origin = f"from_registry({self.force_image})  # forced"
        elif self.spec.docker_image:
            image = modal.Image.from_registry(self.spec.docker_image)
            origin = f"from_registry({self.spec.docker_image})"
        elif self.spec.dockerfile:
            image = modal.Image.from_dockerfile(
                self.spec.dockerfile, context_dir=str(self.spec.dockerfile.parent)
            )
            origin = (
                f"from_dockerfile({self.spec.dockerfile}, "
                f"context_dir={self.spec.dockerfile.parent})"
            )
        else:
            image = modal.Image.from_registry(FALLBACK_IMAGE)
            origin = f"from_registry({FALLBACK_IMAGE})  # task has no docker_image/Dockerfile"

        if self.spec.tests_dir.is_dir():
            # copy=False: mounted at sandbox startup, so no image rebuild per task
            image = image.add_local_dir(str(self.spec.tests_dir), TESTS_MOUNT)
            origin += f" + add_local_dir({self.spec.tests_dir} -> {TESTS_MOUNT})"
        elif self.spec.test_script:
            script = self.task_dir / self.spec.test_script
            image = image.add_local_file(
                str(script), f"{TESTS_MOUNT}/{Path(self.spec.test_script).name}"
            )
            origin += f" + add_local_file({script} -> {TESTS_MOUNT}/)"

        self._image, self.origin = image, origin
        return image

    def build(self):
        """Eagerly build (or fetch the cached) image; timed separately from boot."""
        image = self.image()
        if self.build_seconds is None:
            started = time.perf_counter()
            self._image = image.build(self.app)
            self.build_seconds = time.perf_counter() - started
        return self._image

    def _require_sandbox(self):
        if self.sandbox is None:
            raise RuntimeError(f"{self.name}: reset() before exec()/grade()")
        return self.sandbox

    # -- TaskEnv -----------------------------------------------------------
    def reset(self) -> str:
        self.close()
        image = self.build()
        started = time.perf_counter()
        self.sandbox = self._modal().Sandbox.create(
            "sleep",
            "infinity",
            image=image,
            app=self.app,
            timeout=self.timeout,
            idle_timeout=IDLE_TIMEOUT,
            **self.spec.resources(),
        )
        self.boot_seconds = time.perf_counter() - started
        return self.spec.instruction

    def exec(self, command: str) -> str:
        sandbox = self._require_sandbox()
        try:
            proc = sandbox.exec("bash", "-lc", command, timeout=self.exec_timeout)
            stdout, stderr, _ = _drain(proc)
        except Exception as exc:  # a dead sandbox must not kill the rollout
            return _tail(f"<exec failed: {type(exc).__name__}: {exc}>")
        return _tail(stdout + stderr)

    def grade(self) -> float:
        command = grade_command(self.spec)
        if command is None:
            return 0.0
        sandbox = self._require_sandbox()
        try:
            proc = sandbox.exec("bash", "-lc", command, timeout=self.grade_timeout)
            stdout, stderr, _ = _drain(proc)
        except Exception:
            return 0.0
        return verdict(stdout + stderr)

    def close(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None:
            return
        try:
            sandbox.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Local Docker env (the original bridge behaviour, kept for Linux hosts)
# ---------------------------------------------------------------------------
class LocalDockerTask:
    """One task container as a local `docker run -d` container."""

    def __init__(self, task_dir: str | Path, *, image_tag: str | None = None) -> None:
        self.spec = TaskSpec.load(resolve_task_dir(task_dir))
        self.task_dir = self.spec.task_dir
        self.name = self.spec.name
        self.image_tag = image_tag or f"rl/{self.name.replace('/', '-')}:latest"
        self.exec_timeout = EXEC_TIMEOUT
        self.grade_timeout = GRADE_TIMEOUT
        self.container: str | None = None

    def _docker(self, *args: str, check: bool = True, timeout: int | None = None):
        try:
            proc = subprocess.run(
                ["docker", *args], capture_output=True, text=True, timeout=timeout
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"docker is not installed: {exc}") from exc
        if check and proc.returncode != 0:
            detail = _tail(proc.stdout + proc.stderr, 2000)
            raise RuntimeError(f"docker {args[0]} failed ({proc.returncode}): {detail}")
        return proc

    def image_ref(self) -> str:
        if self.spec.docker_image:
            return self.spec.docker_image
        if self.spec.dockerfile is None:
            raise RuntimeError(
                f"{self.name}: no [environment].docker_image and no environment/Dockerfile"
            )
        self._docker(
            "build", "-q", "-t", self.image_tag, "-f", str(self.spec.dockerfile),
            str(self.spec.dockerfile.parent), timeout=3600,
        )
        return self.image_tag

    def reset(self) -> str:
        self.close()
        image = self.image_ref()
        self.container = self._docker("run", "-d", "--rm", image, "sleep", "infinity").stdout.strip()
        # Harbor mounts the verifier at /tests; a bare `docker run` does not, so copy it in.
        if self.spec.tests_dir.is_dir():
            self._docker("cp", f"{self.spec.tests_dir}{os.sep}.", f"{self.container}:{TESTS_MOUNT}",
                         check=False, timeout=EXEC_TIMEOUT)
        elif self.spec.test_script:
            script = self.task_dir / self.spec.test_script
            self._docker("exec", self.container, "mkdir", "-p", TESTS_MOUNT, check=False)
            self._docker("cp", str(script), f"{self.container}:{TESTS_MOUNT}/{script.name}",
                         check=False, timeout=EXEC_TIMEOUT)
        self._docker("exec", self.container, "bash", "-lc",
                     f"mkdir -p {REWARD_DIR}", check=False)
        return self.spec.instruction

    def exec(self, command: str) -> str:
        if self.container is None:
            raise RuntimeError(f"{self.name}: reset() before exec()")
        try:
            proc = self._docker("exec", self.container, "bash", "-lc", command,
                                check=False, timeout=self.exec_timeout)
        except subprocess.TimeoutExpired as exc:
            return _tail(f"<exec timed out after {self.exec_timeout}s: {exc}>")
        return _tail(proc.stdout + proc.stderr)

    def grade(self) -> float:
        command = grade_command(self.spec)
        if command is None or self.container is None:
            return 0.0
        try:
            proc = self._docker("exec", self.container, "bash", "-lc", command,
                                check=False, timeout=self.grade_timeout)
        except (subprocess.TimeoutExpired, RuntimeError):
            return 0.0
        return verdict(proc.stdout + proc.stderr)

    def close(self) -> None:
        container, self.container = self.container, None
        if container is None:
            return
        try:
            self._docker("rm", "-f", container, check=False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory: pick the backend with RL_ENV=modal|docker (default modal)
# ---------------------------------------------------------------------------
def make_env(task_dir: str | Path, kind: str | None = None) -> TaskEnv:
    """One TaskEnv for one task, chosen by `kind` or the RL_ENV environment variable."""
    backend = (kind or os.environ.get("RL_ENV") or "modal").strip().lower()
    if backend in ("modal", "sandbox"):
        return ModalTask(task_dir)
    if backend in ("docker", "local", "local-docker"):
        return LocalDockerTask(task_dir)
    raise ValueError(f"unknown RL_ENV={backend!r}; expected 'modal' or 'docker'")


# ---------------------------------------------------------------------------
# Smoke test: real sandbox, real task image
# ---------------------------------------------------------------------------
def smoke(task_dir: str | Path, *, run_grader: bool = False) -> int:
    """Prove both sandbox paths work. Costs a few cents of Modal credits."""
    modal = ModalTask._modal()
    print(f"modal version: {modal.__version__}")

    # 1) cheapest possible API check -- no task image, the documented fallback base
    started = time.perf_counter()
    app = _modal_app(modal)
    tiny = modal.Sandbox.create(
        "sleep", "infinity",
        image=modal.Image.from_registry(FALLBACK_IMAGE),
        app=app, timeout=600, idle_timeout=IDLE_TIMEOUT,
    )
    try:
        stdout, stderr, code = _drain(
            tiny.exec("bash", "-lc", "echo hello", timeout=120)
        )
    finally:
        tiny.terminate()
    elapsed = time.perf_counter() - started
    answer = stdout.strip()
    print(f"echo sandbox: {answer!r} (exit {code}, {elapsed:.2f}s wall-clock, image={FALLBACK_IMAGE})")
    if answer != "hello":
        print(f"FAIL: expected 'hello', got {answer!r}; stderr={stderr!r}", file=sys.stderr)
        return 1

    # 2) one real task image: task.toml's docker_image, else the task's Dockerfile
    env = ModalTask(task_dir)
    image = env.image()  # resolve before timing so the label is available
    print(f"task: {env.name}")
    print(f"task image: {env.origin}")
    try:
        env.reset()  # eager image build + `modal.Sandbox.create`
    except Exception as exc:  # exact error, then prove the smoke on the fallback image
        print(f"task image failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"falling back to {FALLBACK_IMAGE}", file=sys.stderr)
        env = ModalTask(task_dir, force_image=FALLBACK_IMAGE)
        env.reset()
    first_exec = float("nan")
    try:
        started = time.perf_counter()
        version = env.exec("python3 --version").strip()
        first_exec = time.perf_counter() - started
        print(f"python3 --version: {version}")
    finally:
        env.close()

    print(f"image build (once, cached afterwards): {env.build_seconds:.2f}s wall-clock")
    print(f"image-backed sandbox creation: {env.boot_seconds:.3f}s wall-clock "
          f"(timeout={env.timeout}s, {env.spec.resources()})")
    print(f"first exec in that sandbox (time-to-first-output): {first_exec:.3f}s wall-clock")

    if run_grader:
        # End-to-end: boot a fresh episode, run one command, then the real verifier.
        env = ModalTask(env.task_dir)  # rebuild the task-proper env (not the fallback)
        env.reset()
        try:
            print(f"exec 'echo hi from $(pwd)': {env.exec('echo hi from $(pwd)').strip()}")
            print(f"grade() on the untouched workspace: {env.grade():.1f} (0.0 expected)")
        finally:
            env.close()

    ok = version.startswith("Python 3")
    print("SMOKE PASSED" if ok else "SMOKE FAILED")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Modal-backed TaskEnv for Harbor tasks")
    ap.add_argument("--smoke", action="store_true",
                    help="boot a sandbox + one real task image (spends Modal credits)")
    ap.add_argument("--grade", action="store_true",
                    help="with --smoke: also run the task's verifier end-to-end")
    ap.add_argument("--task", default="data/train/aider-polyglot/polyglot_python_transpose",
                    help="task directory for --smoke")
    args = ap.parse_args(argv)
    if not args.smoke:
        ap.print_help()
        return 2
    return smoke(args.task, run_grader=args.grade)


if __name__ == "__main__":
    raise SystemExit(main())
