import subprocess
import os

class SandboxRunner:
    def __init__(self, workspace_dir="data/experiments"):
        self.workspace_dir = workspace_dir
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)

    def execute(self, script_code, experiment_id="exp_01"):
        script_path = os.path.join(self.workspace_dir, f"{experiment_id}.py")
        
        with open(script_path, "w") as f:
            f.write(script_code)
            
        print(f"Executing experiment {experiment_id} in isolated process...")
        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=300 # 5 minutes max
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out after 5 minutes."
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e)
            }

if __name__ == "__main__":
    print("SandboxRunner ready.")
