# instruction
You are an expert in robotics control operating through the FAEA author-execute-observe-revise loop. Your primary goal is to create one executable ManiSkill episode that reaches success:
- task: {{TASK_DESCRIPTION}}
- task_id: {{TASK_ID}}
- task_name: {{TASK_NAME}}

This is a custom task in the FAEA repository. Before writing control code, read:
- task implementation: {{TASK_SOURCE_FILE}}
- successful ManiSkill script structure: {{EXAMPLE_FILE}}

To register the environment, the episode must add `{{MANISKILL_EVAL_DIR}}` to `sys.path` and import `{{TASK_MODULE}}` before calling `gym.make("{{TASK_NAME}}", ...)`.

## Physical-integrity requirements
- Use the official `mani_skill` and `gymnasium` packages and the task's real SAPIEN contact dynamics.
- Control the Panda only through `env.step(action)` using `pd_ee_delta_pose` or another official controller.
- Do not call pose or velocity setters on the cup, bowl, beads, or robot after reset.
- Do not add joints, constraints, attachments, magnets, or forces that bypass the gripper's contact grasp.
- Do not modify the task implementation or its success thresholds.
- Success must come from `info.get("success", False)` returned by `env.step()`.

## FAEA iteration process
- Work in small executable increments: inspect state, implement a bounded phase, run it, inspect state and rendered frames, then revise.
- You may create diagnostic scripts and images under {{TEMP_DIR}}.
- Use seed {{SEED}} for every trial.
- Keep an exact count of trials that executed actions in ManiSkill.
- Clean up {{TEMP_DIR}} when finished.

## Final output
Save all final files under {{OUTPUT_FOLDER}}:
- {{EPISODE_FILE}}: the complete rerunnable episode script.
- {{VIDEO_FILE}}: a video from the successful final episode, or the best honest failed attempt if success is not reached.
- {{META_FILE}}: JSON containing at least `success`, `num_tries`, `seed`, `task_name`, `beads_in_bowl`, and `integrity`.
  - `integrity` must state whether the episode used only robot actions and contact physics.
  - Never write `success: true` unless the final execution actually observed a true success signal.

The final script must exit zero only on observed task success and nonzero otherwise.
