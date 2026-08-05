"""Latent Sokoban as a PufferLib environment.

Build the extension first:

    python csrc/puffer/setup.py build_ext --inplace

Then:

    python csrc/puffer/sokoban.py        # throughput benchmark

The observation is the pair the agent protocol serves: a 64x64x3 render of the
current board followed by a 64x64x3 render of the goal, flattened to 24576
bytes. That layout is deliberate -- it is exactly what distill/agent.py and
baseline/agent.py consume from the server, so a policy trained here takes the
same input at evaluation time.
"""

import gymnasium
import numpy as np

import pufferlib

try:
    from csrc.puffer import binding
except ImportError:                       # built in place
    import binding

IMG = 64
OBS_BYTES = 2 * IMG * IMG * 3


class Sokoban(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128,
                 board_size=8, n_crates=1, pool_size=256, max_steps=30,
                 min_len=4, max_len=20, wall_density=10, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(OBS_BYTES,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards, self.terminals,
            self.truncations, num_envs, seed,
            board_size=board_size, n_crates=n_crates, pool_size=pool_size,
            max_steps=max_steps, min_len=min_len, max_len=max_len,
            wall_density=wall_density)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        return (self.observations, self.rewards, self.terminals,
                self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == "__main__":
    import time

    N = 1024
    # A small pool keeps startup quick for a benchmark; training wants more
    # level diversity than this.
    env = Sokoban(num_envs=N, pool_size=64)
    env.reset()

    CACHE = 1024
    actions = np.random.randint(0, 4, (CACHE, N))
    steps = 0
    start = time.time()
    i = 0
    while time.time() - start < 5:
        env.step(actions[i % CACHE])
        steps += N
        i += 1
    print("Sokoban SPS:", int(steps / (time.time() - start)))
    env.close()
