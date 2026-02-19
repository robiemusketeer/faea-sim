# instruction
You are an expert in robotics control. Your primary goal is to write a **policy function** that solves the following LIBERO task:
- Task: {{TASK_DESCRIPTION}}
- task_id: {{TASK_ID}}

## Harness (pre-generated — do NOT modify)
A scaffold has been pre-generated at `{{SCAFFOLD_FILE}}`. **Do NOT modify it.**
It handles env creation, frame recording, success checking, video saving, and metadata saving.

You only need to write the policy file at `{{POLICY_FILE}}`.

Your policy file must define:

```python
import numpy as np

def policy(env, obs, h):
    # ... task-specific action phases ...
    return obs
```

The helpers namespace `h` provides:
- `h.execute_action(env, action, steps=1)` → (obs, done) — steps the env and auto-records frames
- `h.compute_action(current_pos, target_pos, gripper, gain=15)` → 7-d action array
- `h.record_frame(obs)` — manually record an extra frame
- `h.GRIPPER_CLOSE` (1.0) and `h.GRIPPER_OPEN` (-1.0)
- `h.np` — numpy (also available via your own `import numpy as np`)

To run the episode: `python {{SCAFFOLD_FILE}}`

See {{EXAMPLE_FILE}} for a working example of a policy function.

## Requirements
- Do not cheat.
- Do not reference or use outputs from other tasks or trials. Each task must be solved independently.

## Tips
- libero docs: https://lifelong-robot-learning.github.io/LIBERO/html/index.html
- You can create intermediate scripts to help. The intermediate script and artifacts should live under: {{TEMP_DIR}}
- You can follow the ReAct cycle, but limit to 10 tries. If it's over 10 tries, assume failure.
- Keep the policy simple, no need for error checks.
- Do not use any local files, except for those inside your workspace.
- Clean up {{TEMP_DIR}} after each execution.

## Output
Write your policy to: `{{POLICY_FILE}}`
Then run: `python {{SCAFFOLD_FILE}}`

The scaffold automatically saves all outputs to {{OUTPUT_FOLDER}}:
- Video recording of the episode
- Metadata JSON with success result
