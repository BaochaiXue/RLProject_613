import subprocess
from typing import List


def call_script(script_name: str, param: int | None) -> None:
    try:
        if param is None:
            cmd: List[str] = ["python", script_name]
        else:
            cmd: List[str] = ["python", script_name, str(param)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            print(f"{script_name} executed successfully.")
        else:
            print(f"{script_name} execution failed.")
    except Exception as e:
        print(f"An error occurred while running {script_name}: {e}")


if __name__ == "__main__":
    time_step_of_training: int = 1000
    scripts: List[str] = ["environment.py", "other_rl_trains.py", "rl_test.py"]
    call_script(scripts[0], time_step_of_training)
    call_script(scripts[1], time_step_of_training)
    call_script(scripts[2], None)