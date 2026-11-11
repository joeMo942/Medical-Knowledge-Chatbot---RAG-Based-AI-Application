# Target the author's laptop GPU, not a GPU-less grader machine

The brief says graders run the project on "a clean machine with Docker installed". The author decided instead that it will run on their own laptop: RTX 5070 Laptop GPU with 8 GB of VRAM, 31 GB RAM, Core Ultra 9 275HX.

So the default compose file requests the NVIDIA GPU, and model choices are sized to 8 GB of VRAM and a 30 s limit per `/chat` request. A CPU-only override file exists as a documented, slower fallback, but it is not what the models are tuned for. That is why the embedding model runs on the CPU: the GPU stays free for the LLM, so the two models don't swap in and out of the same 8 GB on every request.

## Consequences

- On a machine without an NVIDIA GPU, the default compose file will not start; the CPU override is needed, and responses may exceed 30 s: the routing prompt alone (about 2,700 tokens) can take 10 to 30 s on a CPU. The CPU override therefore widens the time limits: `MEDRAG_BUDGET_S` (25 s on the GPU, 240 s on the CPU) and `MEDRAG_ROUTE_TIMEOUT_S` (4 s on the GPU, 90 s on the CPU), among others; the README section "No NVIDIA GPU" lists the knobs. Only the 8 GB NVIDIA GPU was tested.
- If the project ever has to run on unknown hardware, the model choice in `MODEL_DECISION.md` has to be revisited, not just the compose file.
