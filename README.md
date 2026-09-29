# AquaWAM: A Dynamics-aware World Action Model for Underwater Embodied Agents

**Project page:** https://cunhaozhu.github.io/AquaWAM/

**Paper:** https://arxiv.org/abs/2609.33299

AquaWAM is the first World Action Model designed for underwater embodied agents. Underwater vehicles keep moving after a command ends, because of inertia, buoyancy, drag, and currents. Existing world action models mostly predict future images, so they do not capture this passive motion.

AquaWAM predicts a compact physical state instead of future images. It models the thruster dead band, the glide that outlasts each command, and ambient currents, then chooses the action whose predicted future is better.

On the USIM benchmark, AquaWAM reaches a 72.6% success rate over 20 underwater tasks, and makes action decisions 2.7x faster than U0 on an NVIDIA Jetson AGX Orin. Without DVL velocity, it still reaches 61.6%, compared with 39.4% for U0.

Code is coming soon.

- Dataset: https://huggingface.co/datasets/Vincent2025hello/usim
- Environment: https://github.com/VincentGu2000/u0env
