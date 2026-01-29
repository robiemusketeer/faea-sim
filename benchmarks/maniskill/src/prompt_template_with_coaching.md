# instruction
You are an expert in robotics control. Your primary goal is to create a single script, containing an episode that reaches success:
- with task: {{TASK_DESCRIPTION}}
- task_id: {{TASK_ID}}
- task_name: {{TASK_NAME}}

## Requirements
- The final script:
    - must check success via `info.get('success', False)` from env.step() - note: success may be a tensor, use `.item()` if needed
    - contains a sequence of actions
    - use official mani_skill package
    - user can rerun with micromamba env maniskill when invoked
- Do not cheat.
- Do not reference or use outputs/files from other tasks or trials. Each task must be solved independently, except for those inside your workspace.
    - One exception: you can reference /home/btsui/startup/farm/farm-experiment/faea/maniskill_eval/example/pick_and_place_simple.py, a successful example of pick and place, as a template of how an episode looks.


## Instruction
- ManiSkill docs: https://maniskill.readthedocs.io/
- You can create intermediate scripts to help. The intermediate script and artifacts should live under: {{TEMP_DIR}}
- You can follow the ReAct cycle.
- keep the final script simple, no need for error checks.
- clean up {{TEMP_DIR}} after each execution

## Output
All output files must be saved to: {{OUTPUT_FOLDER}}
- {{META_FILE}}: metadata of this episode
    - Contains:
        - success: {True,False}
        - num_tries: N tries on the ManiSkill execution with actions
- {{EPISODE_FILE}}: Output final script
- {{VIDEO_FILE}}: Video recording of the episode

## Tips on building out the episode script:
- Think about how a human would accomplish the task. Break it down into a sequence of descriptive actions.
- Build the episode iteratively from small chunks of actions in each ReAct cycle. Couple pointers:
    - Record the current state (e.g. image, states of objects).
    - Plan and execute the next sequence of movements you feel confident
    - Review to make sure it's moving according to plan (e.g. review states of objects, image capture, etc.).
- Keep a high z height of the gripper when moving around to avoid hitting things
- Your gripper is very wide in one direction. Try rotate gripper when the gripper keeps hitting something unintended.
- Beware of obstacle in the action trajectory
