"""gpu-cosplay: make a beefy GPU pretend to be a smaller one.

The host-side package lives outside the container. Inside the container, a tiny
sibling helper (`gpu_cosplay.inject`) is shipped in the image to apply the VRAM
cap at process start.
"""

__version__ = "0.1.0"
