import subprocess

subprocess.run([
    "python3", "-m", "run_reconstruction",
    "-c", "pretrained/E2VID_lightweight.pth.tar",
    "-i", "data/dynamic_6dof.zip",
    "--auto_hdr",
    "--display",
    "--show_events"
])

#RESULTS
#1s: video very blurry. Hard time reconstructing somewhat complex surfaces such as faces and tables
#0.1s: original dvs video a bit blurry. Reconstruction pretty good but laggy due to low fps(10)
#0.01s: both videos good. Probably best. While slow or nearly no movement image becomes white due to missing information for the pixel in this event frame.
#0.001s: original video barely okay. Reconstruction just grey blur. Not enough information per event_frame for reconstruction.
