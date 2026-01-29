# instruction
You are an expert in robotics control. Your primary goal is to create a single script, containing an episode that reaches success:
- with task: {{TASK_DESCRIPTION}}
- task_id: {{TASK_ID}}

## Requirements
- The final script:
    - must use success() within Libero Env, as the final metric of success
    - contains a sequence of actions
    - use official libero package within libero named "libero"
    - user can rerun with micromamba env libero when invoked
- Do not cheat.
- Do not reference or use outputs/files from other tasks or trials. Each task must be solved independently, except for those inside your workspace.
    - One exception: you can reference {{EXAMPLE_FILE}}, a successful example of pick and place, as a template of how an episode look.


## Instruction
- libero docs: https://lifelong-robot-learning.github.io/LIBERO/html/index.html
- You can create intermediate scripts to help. The intermediate script and artifacts should live under: {{TEMP_DIR}}
- You can follow the ReAct cycle.
- keep the final script simple, no need for error checks.
- clean up {{TEMP_DIR}} after each execution

## Output
All output files must be saved to: {{OUTPUT_FOLDER}}
- {{META_FILE}}: metadata of this episode
    - Contains:
        - success: {True,False}
        - num_tries: N tries on the libero execution with actions
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
