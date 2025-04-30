import subprocess

def get_gpu_name():
    try:
        line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        line = line_as_bytes.decode("ascii")
        _, line = line.split(":", 1)
        line, _ = line.split("(")
        return line.strip()
    except subprocess.CalledProcessError:
        return "N/A"

gpu_name = get_gpu_name()
print(f"GPU Name: {gpu_name}")