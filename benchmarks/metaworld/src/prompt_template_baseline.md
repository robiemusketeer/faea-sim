# instruction
You are an expert in robotics control. Your primary goal is to create a single script, containing an episode that reaches success:
- with task: {{TASK_DESCRIPTION}}
- task_id: {{TASK_ID}}

## Requirements
- The final script:
    - must use info['success'] from the Metaworld environment as the final metric of success
    - contains a sequence of actions
    - use official metaworld package with gymnasium
    - user can rerun with micromamba env metaworld when invoked
- Do not cheat. Do not try to find the solution by looking up the solution. Do not read anything under path metaworld/policies
- Do not reference or use outputs/files from other tasks or trials. Each task must be solved independently, except for those inside your workspace.
    - One exception: you can reference /home/btsui/startup/farm/farm-experiment/faea/metaworld_eval/example/pick_place_example.py, a successful example of pick and place, as a template of how an episode look.

## Instruction
- Metaworld docs: https://github.com/Farama-Foundation/Metaworld
- You can create intermediate scripts to help. The intermediate script and artifacts should live under: {{TEMP_DIR}}
- You can follow the ReAct cycle.
- keep the final script simple, no need for error checks.
- clean up {{TEMP_DIR}} after each execution
- fail the trial if it has been running for more than 30 minutes

## Output
All output files must be saved to: {{OUTPUT_FOLDER}}
- {{META_FILE}}: metadata of this episode
    - Contains:
        - success: {True,False}
        - num_tries: N tries on the metaworld execution with actions
- {{EPISODE_FILE}}: Output final script
- {{VIDEO_FILE}}: Video recording of the episode
