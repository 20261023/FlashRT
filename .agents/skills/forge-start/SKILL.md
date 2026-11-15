---
name: forge-start
description: Configure and launch a ForgeKernel operator optimization - pick a local or ssh GPU, pick the reference, check the environment, and resume existing runs. Use for "launch ForgeKernel", "optimize this operator with ForgeKernel", "continue my ForgeKernel run"; a request for a design or a preflight only does not start an optimization.
---

# ForgeKernel one-step launch

Turn the user's operator goal into a runnable task and drive it through the repository's own `forge` lifecycle. When the user has asked to launch and everything needed is known, go ahead; ask for whatever is missing in one batch, and never ask again for an authorization already given.

## Locate the repository and the authoritative configuration

Find ForgeKernel in the directory the user names, or the current directory; check its git status, commit and remote. With no checkout, clone https://github.com/OpenBMB/ForgeKernel.git. Never overwrite existing changes, and never switch the version an existing run uses.

Read that version's README.md (README_zh.md for Chinese), local.env.example, the task's task.toml and global_config/ first; docs/design/v1-launch.md when needed. They are the authority on configuration: this skill fixes no model, budget or dependency version, with one exception - the loop's wall-clock `hours` defaults to 24 (item 4 below).

This skill is the user-facing launcher for Claude Code. Install it in the personal skill directory `~/.claude/skills/forge-start/` (or a project's `.claude/skills/forge-start/`). Never put it under the repository's `skills/`: that directory is mirrored to the internal teacher and student agents.

## Collect what is missing, once

Read the existing task, the machine configuration and this conversation first; ask only what cannot be inferred:

1. What to optimize: an existing task directory, or the operator's signature, semantics, shapes, dtypes and layouts. Never invent the semantics or the legal input ranges.
2. Where and on which GPU: this machine's GPU, an ssh host the user already has, or a cloud resource still to be requested? Which GPU model and device index? The model can be detected if unknown; on a multi-GPU machine never pick a device the user did not name.
3. Whom to beat: the user's implementation, an expert library, a PyTorch implementation, or no reference at all? Get the source path and the callable entry point; a performance number alone is not a runnable reference.
4. How long: use the budget the user gives for this launch; otherwise `hours = 24`, overriding the `hours` the task's task.toml sets (the rmsnorm example included) and the global_config value. Write it into the task.toml of a task this launch creates; for an existing task directory, leave the original untouched and write it into a task variant. Do not change global_config. `max_meta` and `reference_hours` keep their merged values. Show the final `hours` and `max_meta` (plus `reference_hours` when there is no reference) with where each comes from, and ask the user to confirm or change them; if the user gives nothing, use the defaults above, but say so in the summary, and say that the loop also stops before 24 hours once the teacher rounds reach `max_meta`.

When no operator is named and the user only wants to try ForgeKernel, use the repository's rmsnorm example. When an operator is named but its semantics, shapes or dtypes are missing, keep asking: never substitute the example, and never assume typical shapes.

## Resource discovery

This skill stores no concrete cluster, resource pool, image, account, billing project, host address or shared-disk path, and never treats another skill's past examples as the current configuration.

- This machine: ask for or detect the GPU model, and settle the device index.
- An existing remote: read the ssh host and connection settings the user provides; verify it is reachable and which GPU it really has.
- A resource still to be requested: confirm the platform first, then query that platform's own tools live for the clusters, GPUs, quotas, resource pools, images and projects the user can use. If the query fails, a choice cannot be inferred, or connection details are missing, ask the user for them in one batch. Honor authorizations already given; do not ask again.

Choose local or ssh delivery from the real environment; what is supported is whatever the current repository supports. The delivery directory must be an absolute path dedicated to this launch, so no user code or shared data is overwritten. Resource discovery and dependency deployment are separate steps; the deploy script never creates cloud resources.

## Locate the skill directory

This skill supports Claude Code only. Below, `$SKILL_DIR` is the directory holding this SKILL.md: use the base directory Claude Code gives when it loads the skill; failing that, look in the project and then the personal skill directory, and stop at the first one that contains `scripts/start.py`:

```bash
for d in "$PWD/.claude/skills/forge-start" "$HOME/.claude/skills/forge-start"; do
  [ -f "$d/scripts/start.py" ] && SKILL_DIR="$d" && break
done
```

If neither has it, stop and ask the user where the skill is installed; do not guess.

## The one-step launch entry point

When the user asks to launch, the final entry point is [scripts/start.py](scripts/start.py), not deploy.py. After resource discovery and the task and local.env configuration, run it once:

```bash
python3 "$SKILL_DIR/scripts/start.py" \
  --repo "$REPO" --env "$VENV" --role controller \
  --task "$TASK" --session "$SESSION_FILE" --claude "$CLAUDE_BIN"
```

With the GPU on the same machine use `--role both`; the other install options are the same as deploy.py's. For an ssh GPU host, deploy the remote side first: wait until the source transfer and the deploy command have exited successfully before running start.py, never concurrently with it; the user installs nothing by hand. The entry point deploys the controller environment, checks the real GPU architecture against the task, validates the configuration, creates the run and calls the real `forge run`. Authentication and the sandbox are verified and enforced by ForgeKernel's real agent turns; build no separate policy.

The session file holds only a task-to-run pointer; the run's state is whatever ForgeKernel says. Repeating the same command resumes the same run directory; a file lock stops two launches of one session. On failure, follow the real error, fix the environment or configuration and retry; a successful install or a `FORGE_RUN` line is no proof of acceptance. Report a full launch as passed only after observing artifacts of the actual optimization phase and evidence of GPU execution; while model networking, a hardware mismatch or the like is unresolved, label it explicitly as not accepted.

## One-step dependency deployment

Run [scripts/deploy.py](scripts/deploy.py), shipped with this skill; read its `--help` first. It runs on the target machine with role `controller` (the controlling machine), `gpu` (the GPU machine) or `both` (the same machine). The agent fills in the arguments from the chosen task and target environment; the user installs no dependency by hand.

```bash
python3 "$SKILL_DIR/scripts/deploy.py" \
  --repo "$REPO" --env "$VENV" --role controller --claude "$CLAUDE_BIN"
python3 "$DEPLOY_SCRIPT" \
  --repo "$REMOTE_REPO" --env "$REMOTE_VENV" --role gpu \
  --dsl-version "$DSL_VERSION"
```

For ssh deployment, transfer the script and the source first, then run it on the remote; all paths come from this launch's discovery. The script reuses a valid virtual environment. A new controller environment is installed standalone; a GPU environment inherits the CUDA packages already in the image, and gets ForgeKernel plus the dependencies of its role; the GPU role installs no agent. If CUDA PyTorch is missing, choose `--torch-spec` and the package index from the real driver; never blindly install the default CPU build. The CuTe DSL version comes from the task's environment; do not pass it for CUDA C++ tasks.

- Online: detect a reachable `--index-url` and `--npm-registry`. With Claude Code already installed, pass `--claude`; without it, the script installs it with npm into a dedicated directory, and installs Node/npm into the dedicated virtual environment when they are missing.
- Offline: on a networked machine, prepare wheels for the target OS, architecture and Python ABI (build and transitive dependencies included), bring them to the target and pass `--wheelhouse`. The controller also needs `--claude` pointing at an executable for its platform; nothing is downloaded online and no login cache is copied.
- A macOS controller with a Linux GPU: Linux extensions cannot be imported on macOS. Pass `--dsl-source-wheels` to start.py, listing the Linux CuTe DSL wheel and its library wheels at the GPU's version. The script unpacks them into the virtual environment's dsl-source and sets up source discovery, without importing Linux extensions on the controller; GPU extensions run on the GPU machine only.
- `--dry-run` only prints the install plan: it runs no command and creates no directory. An install failure stops immediately; no version is swapped and no system setting changed automatically. The script's `dependencies_ready` output means only that the dependencies and their checks passed, not that authentication, the sandbox or the optimization succeeded.

## Runtime environment and sandbox boundary

The sandbox belongs to the agent runtime, ForgeKernel's policy and the host platform. The skill and the deploy script implement no sandbox, and neither install nor change isolation policies, container permissions or kernel settings.

After deployment, verify separately the model authentication, real GPU computation and the sandbox capabilities the runtime requires. Read the agent CLI's current help before using it. Report authentication status only, never credentials. If the runtime sandbox is unavailable, report the exact missing tool or platform limit and hand it to the environment setup, or choose a compatible controller with an ssh GPU; never disable or loosen the sandbox to claim a successful launch.

On the real GPU, check the model, the compute capability and task.arch; when they differ, create a task variant rather than overwrite the original task. Check the compiler, ncu and the tools reference-free generation needs, per task; without ncu, fall back to the timing-only path only where the repository supports it, and state the limitation.

## Choosing the reference

Settle whom the user wants to beat first, then decide source visibility:

- Production optimization: prefer the user's current implementation.
- Beating an expert library: choose a concrete implementation compatible in semantics, precision, shapes and target GPU; record its source and version, and never automatically pick whatever the web claims is fastest. Verify sources when searching.
- PyTorch only: usable as the initial reference, but never describe beating PyTorch as beating an expert kernel.
- No reference at all: write no reference table and use the repository's generation flow; show the reference budget and the optimization budget separately, and never call the loop budget a hard cap on the whole flow.

When the source is readable and the teacher may read it: `reference.run` + `reference.source`, source mode. With only a callable entry point, or when the source may not be read: write only `run`, or set `blackbox = true` as the task conventions say. Never expose reference source to the student or bypass the harness's isolation.

Check the call signature, mathematical semantics, dtypes, layouts, shapes, legal inputs and dependencies. The harness derives the correctness criterion from the independent operator description and calibrates the reference against it; the reference's behavior is not automatically the definition of correct. The reference and candidates are timed in the same harness job; never treat paper or historical numbers as measured baselines.

## Configure, launch and resume

Machine settings and credentials go in local.env or environment variables; task differences in task.toml; defaults come from global_config/. Keep existing valid settings, and remember that environment variables override local.env. Never write credentials into a task, a report or the repository.

Show a short summary first: the task, the GPU's location and model, the reference's source and mode, the target, the model and the effective budget. When the user asked for a preflight only, create no run and start no optimization.

Use the `forge` of the chosen environment; read `forge --help` and the relevant subcommand help before running:

```bash
forge new tasks/<op> --dry-run
forge new tasks/<op>
forge run .artifacts/<run>
forge status .artifacts/<run>
```

A dry run only validates the configuration and prints the plan; it proves nothing about ssh, the GPU, authentication or whether the model can be called. Create and run only after it succeeds and the real environment checks pass. When the user asked to launch, continue into the actual optimization phase; never stop after dependency deployment or a dry run and claim a successful launch. Use the exact run directory `forge new` returns; never guess it from file times.

When the user explicitly asks for a from-scratch test, create a new, not-yet-existing controller directory, virtual environment, GPU working directory, task copy and session file; never read old runs or best kernels, or reuse old compile caches. Already-downloaded install packages and a clean base image may be used, but state that boundary. Acceptance must cover real model calls, reference calibration, optimization, confirmation and the final report.

To resume, read the named run's status first, then `forge run` the same directory; never `forge new` again. For a repeated launch request, check existing runs first; if several candidates exist and the right one cannot be told, ask the user. Create a new run for a finished task only when the user explicitly asks for a rerun. Verify whether the process or job is still running; never relaunch on the strength of a possibly stale record alone.

Follow the current host's rules for long-running work. Where background tasks are forbidden, run in the foreground, and do not get around it with nohup, shell `&` or a detached tmux; when the host cannot carry the run, report the blocker and the exact resume command, and never claim it has started. On a run error, stop the current step at once, keep the real error and the run directory, and never retry endlessly or quietly change the target to hide a failure.

## Acceptance and output

Distinguish "configuration check passed", "device check passed", "running", "target reached" and "finished without reaching the target". Judge from that version's exit semantics, status, report.md and the final confirmation; a created process or an existing best/kernel.py is not success.

Check the final confirmation's timing backend, host-bound warnings and sampled nodes. When host submission overhead dominates, report only the measured ratio for that window, never a pure GPU kernel speedup; repeated confirmations on one GPU say nothing about stability across nodes. Record the real end-to-end time; the loop budget is no substitute for the end-to-end time, which includes deployment, finishing the current round and the final confirmation.

Output the commit, the real GPU, the run directory, the current phase, absolute paths of the available report and the best kernel, and the status and resume commands. If only a CPU smoke test was done, say plainly that the GPU, model authentication and the full optimization were not verified.
