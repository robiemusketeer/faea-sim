#!/usr/bin/env python3
"""
Systematically execute Claude Agent SDK on Metaworld benchmark tasks.

Usage:
    # List available benchmarks
    python run_metaworld_claude.py --list-benchmarks

    # List tasks in a benchmark
    python run_metaworld_claude.py --list-tasks --benchmark metaworld_all

    # Run a specific task (requires --experiment)
    python run_metaworld_claude.py --experiment exp_001 --task-id 40

    # Run all tasks (requires --experiment)
    python run_metaworld_claude.py --experiment exp_001 --all

Requirements:
    pip install claude-agent-sdk
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Any, List, Tuple

from tqdm import tqdm

# Claude Agent SDK imports
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ThinkingBlock,
)


# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
METAWORLD_EVAL_DIR = SCRIPT_DIR.parent
DATA_DIR = METAWORLD_EVAL_DIR / "data"
TASKS_CONFIG = SCRIPT_DIR / "metaworld_tasks.json"
PROMPT_TEMPLATE_PATH = SCRIPT_DIR / "prompt_template_baseline.md"
EXAMPLE_FILE = (METAWORLD_EVAL_DIR.parent / "examples" / "metaworld_pick_place.py").resolve()

# Default working directory for claude agent
AGENT_CWD = str(METAWORLD_EVAL_DIR)


def list_benchmarks() -> dict:
    """List available benchmarks from config."""
    with open(TASKS_CONFIG) as f:
        return json.load(f)


def get_metaworld_tasks(benchmark_name: str = "metaworld_all") -> List[Tuple[int, str]]:
    """Get tasks from JSON config file."""
    with open(TASKS_CONFIG) as f:
        config = json.load(f)

    if benchmark_name not in config:
        available = list(config.keys())
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Available: {available}")

    task_list = config[benchmark_name]["task_list"]
    return [(task["id"], task["name"]) for task in task_list]


def load_prompt_template(custom_path: str | None = None) -> str:
    """Load the prompt template from the specified or default path."""
    template_path = Path(custom_path) if custom_path else PROMPT_TEMPLATE_PATH
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text()


def format_prompt(template: str, task_id: int, task_name: str, output_dir: Path, benchmark_name: str) -> str:
    """
    Format the prompt template with task_id, task_name, and output directory.

    Uses {{PLACEHOLDER}} syntax for clear template markers.
    """
    # Convert task name from kebab-case to readable format
    readable_task = task_name.replace("-", " ").replace("v3", "").strip()

    # Generate unique temp directory for this task to prevent cross-talk
    temp_dir = f"/tmp/metaworld_sim_{uuid.uuid4().hex[:8]}"

    replacements = {
        "{{TASK_DESCRIPTION}}": readable_task,
        "{{TASK_ID}}": str(task_id),
        "{{ENV_NAME}}": task_name,
        "{{BENCHMARK_NAME}}": benchmark_name,
        "{{META_FILE}}": f"meta_{task_id}.json",
        "{{EPISODE_FILE}}": f"episode_{task_id}.py",
        "{{VIDEO_FILE}}": f"episode_{task_id}.mp4",
        "{{OUTPUT_FOLDER}}": str(output_dir.resolve()),
        "{{TEMP_DIR}}": temp_dir,
        "{{EXAMPLE_FILE}}": str(EXAMPLE_FILE),
    }

    formatted = template
    for placeholder, value in replacements.items():
        formatted = formatted.replace(placeholder, value)

    return formatted


def serialize_message(msg: Any) -> dict:
    """Serialize a message to a JSON-compatible dict."""
    try:
        if hasattr(msg, '__dataclass_fields__'):
            return asdict(msg)
        elif hasattr(msg, '__dict__'):
            return {k: serialize_message(v) for k, v in msg.__dict__.items()}
        else:
            return str(msg)
    except Exception:
        return str(msg)


def validate_task_outputs(output_dir: Path, task_id: int) -> dict:
    """Check if Claude created the expected output files."""
    expected_files = {
        "meta_file": output_dir / f"meta_{task_id}.json",
        "episode_file": output_dir / f"episode_{task_id}.py",
        "video_file": output_dir / f"episode_{task_id}.mp4",
    }
    return {
        name: path.exists()
        for name, path in expected_files.items()
    }


def get_task_success_from_meta(output_dir: Path, task_id: int) -> bool | None:
    """Parse meta file to get actual task success."""
    meta_file = output_dir / f"meta_{task_id}.json"
    if not meta_file.exists():
        return None
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        return meta.get("success", None)
    except Exception:
        return None


def check_task_already_completed(output_dir: Path, task_id: int) -> bool:
    """Check if a task has already been completed (for resume mode)."""
    messages_file = output_dir / f"messages_{task_id}.json"
    if not messages_file.exists():
        return False
    try:
        with open(messages_file) as f:
            data = json.load(f)
        return data.get("agent_completed", False)
    except Exception:
        return False


def save_incremental_summary(
    output_dir: Path,
    task_id: int,
    result: dict,
    experiment: str,
    benchmark: str,
    model: str,
    total_tasks: int,
    completed_count: int,
    skipped_count: int,
) -> None:
    """Save incremental summary after each task completes."""
    summary_file = output_dir / f"run_summary_{task_id}.json"
    summary = {
        "experiment": experiment,
        "benchmark": benchmark,
        "model": model,
        "total_tasks": total_tasks,
        "completed_so_far": completed_count,
        "skipped": skipped_count,
        "last_task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "last_result": {
            "task_id": result["task_id"],
            "agent_completed": result["agent_completed"],
            "task_success": result["task_success"],
            "output_files": result.get("output_files"),
            "metadata": result.get("metadata"),
            "error": result.get("error"),
        }
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


async def run_claude_agent(
    prompt: str,
    task_id: int,
    output_dir: Path,
    model: str = "claude-opus-4-5-20251101",
    log_file: Path | None = None,
) -> dict:
    """
    Run Claude agent with the given prompt using the Python SDK.

    Returns:
        dict with 'agent_completed', 'task_success', 'messages', 'error' keys
    """
    output_file = output_dir / f"messages_{task_id}.json"

    def log(msg: str):
        """Write to log file if provided, otherwise stdout."""
        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        else:
            print(msg, flush=True)

    log(f"\n{'='*60}")
    log(f"Running Claude Agent for task {task_id}")
    log(f"Output will be saved to: {output_file}")
    log(f"{'='*60}")

    result = {
        "task_id": task_id,
        "agent_completed": False,
        "task_success": None,
        "output_files": {},
        "messages": [],
        "session_id": None,
        "metadata": {},
        "error": None,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
    }

    try:
        options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            model=model,
            cwd=AGENT_CWD,
        )

        messages = []

        async for message in query(prompt=prompt, options=options):
            messages.append(message)

            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ThinkingBlock):
                        thinking = block.thinking
                        if len(thinking) > 500:
                            thinking = thinking[:500] + "..."
                        log(f"  [Thinking]: {thinking}")
                    elif isinstance(block, TextBlock):
                        text = block.text[:200] + "..." if len(block.text) > 200 else block.text
                        log(f"  [Text]: {text}")
                    elif isinstance(block, ToolUseBlock):
                        log(f"  [Tool]: {block.name}")
                        input_str = json.dumps(block.input, indent=4, default=str)
                        indented = "\n".join("    " + line for line in input_str.split("\n"))
                        log(f"  [Input]:\n{indented}")
                    elif isinstance(block, ToolResultBlock):
                        status = "ERROR" if block.is_error else "OK"
                        content = block.content
                        if isinstance(content, str) and len(content) > 500:
                            content = content[:500] + "..."
                        elif isinstance(content, list):
                            content = json.dumps(content, default=str)[:500] + "..."
                        log(f"  [ToolResult {status}]: {content}")
            elif isinstance(message, ResultMessage):
                result["session_id"] = message.session_id
                duration_min = message.duration_ms / 60000.0 if message.duration_ms else 0
                usage = message.usage or {}
                result["metadata"] = {
                    "duration_min": round(duration_min, 2),
                    "cost_usd": message.total_cost_usd,
                    "num_turns": message.num_turns,
                    "is_error": message.is_error,
                    "usage": usage,
                }
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                log(f"  [Result]: Session={message.session_id}, "
                    f"Duration={duration_min:.2f}min, Cost=${message.total_cost_usd}, "
                    f"Tokens(in={input_tokens}, out={output_tokens})")

        result["end_time"] = datetime.now().isoformat()
        result["messages"] = [serialize_message(m) for m in messages]
        result["agent_completed"] = True

    except Exception as e:
        result["error"] = str(e)
        result["end_time"] = datetime.now().isoformat()
        log(f"  [Error]: {e}")

    result["output_files"] = validate_task_outputs(output_dir, task_id)
    result["task_success"] = get_task_success_from_meta(output_dir, task_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log(f"Result saved to: {output_file}")
    log(f"Agent completed: {result['agent_completed']}")
    log(f"Task success: {result['task_success']}")
    log(f"Output files: {result['output_files']}")
    if result.get("error"):
        log(f"Error: {result['error']}")

    return result


async def run_single_task(
    task_id: int,
    task_name: str,
    template: str,
    output_dir: Path,
    model: str,
    experiment: str,
    benchmark: str,
    total_tasks: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    resume: bool,
) -> Tuple[dict | None, bool]:
    """Run a single task with semaphore limiting concurrency."""
    async with semaphore:
        if resume and check_task_already_completed(output_dir, task_id):
            pbar.update(1)
            return None, True

        log_file = output_dir / f"log_{task_id}.txt"
        prompt = format_prompt(template, task_id, task_name, output_dir, benchmark)

        result = await run_claude_agent(
            prompt=prompt,
            task_id=task_id,
            output_dir=output_dir,
            model=model,
            log_file=log_file,
        )

        save_incremental_summary(
            output_dir=output_dir,
            task_id=task_id,
            result=result,
            experiment=experiment,
            benchmark=benchmark,
            model=model,
            total_tasks=total_tasks,
            completed_count=0,
            skipped_count=0,
        )

        pbar.update(1)
        return result, False


async def run_all_tasks(
    tasks_to_run: List[Tuple[int, str]],
    template: str,
    output_dir: Path,
    model: str = "claude-opus-4-5-20251101",
    resume: bool = False,
    experiment: str = "",
    benchmark: str = "",
    parallel: int = 1,
) -> Tuple[List[dict], int]:
    """Run all tasks in parallel with progress tracking."""
    semaphore = asyncio.Semaphore(parallel)

    pbar = tqdm(
        total=len(tasks_to_run),
        desc="Running tasks",
        unit="task",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    coroutines = [
        run_single_task(
            task_id=task_id,
            task_name=task_name,
            template=template,
            output_dir=output_dir,
            model=model,
            experiment=experiment,
            benchmark=benchmark,
            total_tasks=len(tasks_to_run),
            semaphore=semaphore,
            pbar=pbar,
            resume=resume,
        )
        for task_id, task_name in tasks_to_run
    ]

    results_with_skip = await asyncio.gather(*coroutines)

    pbar.close()

    results = [r for r, skipped in results_with_skip if not skipped]
    skipped_count = sum(1 for _, skipped in results_with_skip if skipped)

    return results, skipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Execute Claude Agent SDK on Metaworld benchmark tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List benchmarks
    python run_metaworld_claude.py --list-benchmarks

    # List tasks in a benchmark
    python run_metaworld_claude.py --list-tasks --benchmark metaworld_all

    # Run a specific task
    python run_metaworld_claude.py --experiment my_exp --task-id 40

    # Run all tasks
    python run_metaworld_claude.py --experiment my_exp --all --benchmark metaworld_all
        """
    )

    parser.add_argument("--task-id", type=int, help="Specific task ID to run")
    parser.add_argument("--all", action="store_true", help="Run all tasks in the benchmark")
    parser.add_argument("--benchmark", type=str, default="metaworld_all", help="Metaworld benchmark to use")
    parser.add_argument("--list-tasks", action="store_true", help="List all tasks in the benchmark and exit")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available benchmarks and exit")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--model", type=str, default="claude-opus-4-5-20251101", help="Model to use")
    parser.add_argument("--resume", action="store_true", help="Resume experiment")
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--prompt-template", type=str, default=None, help="Path to custom prompt template")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output base directory")

    args = parser.parse_args()

    if args.list_benchmarks:
        benchmarks = list_benchmarks()
        print("\nAvailable benchmarks:")
        for name, info in benchmarks.items():
            print(f"  {name}: {info['description']} ({info['tasks']} tasks)")
        return 0

    if args.list_tasks or args.task_id is not None or args.all:
        print(f"Loading tasks from benchmark: {args.benchmark}")
        tasks = get_metaworld_tasks(args.benchmark)

    if args.list_tasks:
        print(f"\nAvailable tasks in {args.benchmark}:")
        for task_id, task_name in tasks:
            print(f"  {task_id}: {task_name}")
        return 0

    if args.task_id is None and not args.all:
        parser.error("Must specify --task-id, --all, --list-tasks, or --list-benchmarks")

    if not args.experiment:
        parser.error("--experiment is required when running tasks")

    if args.output_dir:
        output_dir = Path(args.output_dir) / args.experiment
    else:
        output_dir = DATA_DIR / args.experiment
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume and not args.force:
            print(f"Error: Experiment folder '{output_dir}' already exists with files.")
            print("Use --resume to continue from where you left off, or --force to overwrite.")
            return 1
        if args.force and not args.resume:
            print(f"Warning: --force specified, will overwrite existing results.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {args.experiment}")
    print(f"Output directory: {output_dir}")
    if args.resume:
        print("Resume mode: will skip tasks with existing successful results")

    template_source = args.prompt_template if args.prompt_template else str(PROMPT_TEMPLATE_PATH)
    print(f"Loading prompt template from: {template_source}")
    template = load_prompt_template(args.prompt_template)
    template_copy_path = output_dir / "prompt_template.md"
    with open(template_copy_path, "w") as f:
        f.write(template)
    print(f"Prompt template saved to: {template_copy_path}")

    if args.task_id is not None:
        valid_ids = [t[0] for t in tasks]
        if args.task_id not in valid_ids:
            print(f"Error: task_id {args.task_id} not found. Valid IDs: {valid_ids}")
            return 1
        tasks_to_run = [(args.task_id, dict(tasks)[args.task_id])]
    else:
        tasks_to_run = tasks

    print(f"Tasks to run: {len(tasks_to_run)}")
    print(f"Parallel workers: {args.parallel}")

    results, skipped_count = asyncio.run(run_all_tasks(
        tasks_to_run=tasks_to_run,
        template=template,
        output_dir=output_dir,
        model=args.model,
        resume=args.resume,
        experiment=args.experiment,
        benchmark=args.benchmark,
        parallel=args.parallel,
    ))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    agent_completed = sum(1 for r in results if r["agent_completed"])
    task_succeeded = sum(1 for r in results if r["task_success"] is True)
    print(f"Agent completed: {agent_completed}/{len(results)}")
    print(f"Task succeeded: {task_succeeded}/{len(results)}")
    if skipped_count > 0:
        print(f"Skipped (resume): {skipped_count}")

    for r in results:
        agent_status = "OK" if r["agent_completed"] else "FAIL"
        task_status = "PASS" if r["task_success"] is True else ("FAIL" if r["task_success"] is False else "N/A")
        error_msg = f" - {r.get('error', '')}" if r.get("error") else ""
        print(f"  [Agent:{agent_status}][Task:{task_status}] Task {r['task_id']}{error_msg}")

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0
    for r in results:
        metadata = r.get("metadata", {})
        usage = metadata.get("usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_cost_usd += metadata.get("cost_usd", 0) or 0

    summary_file = output_dir / "run_summary.json"
    summary = {
        "experiment": args.experiment,
        "benchmark": args.benchmark,
        "model": args.model,
        "total_tasks": len(tasks_to_run),
        "agent_completed": agent_completed,
        "task_succeeded": task_succeeded,
        "skipped": skipped_count,
        "timestamp": datetime.now().isoformat(),
        "total_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "cost_usd": round(total_cost_usd, 4),
        },
        "results": [
            {
                "task_id": r["task_id"],
                "agent_completed": r["agent_completed"],
                "task_success": r["task_success"],
                "output_files": r.get("output_files"),
                "session_id": r.get("session_id"),
                "metadata": r.get("metadata"),
                "error": r.get("error")
            }
            for r in results
        ]
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    return 0 if agent_completed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
