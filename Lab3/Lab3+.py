import subprocess

subprocess.run([
    "python3", "-m", "run_reconstruction",
    "-c", "pretrained/E2VID_lightweight.pth.tar",
    "-i", "data/dynamic_6dof.zip",
    "--auto_hdr",
    "--display",
    "--show_events",
    "--fixed_duration",
    "-T", "1"
])

