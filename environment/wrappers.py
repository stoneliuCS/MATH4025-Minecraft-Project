import time
import logging
import gym

from model.environment import create_environment

logger = logging.getLogger(__name__)


class RobustResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        env_name: str,
        max_retries=3,
    ):
        super().__init__(env)
        self.max_retries = max_retries
        self.env_name = env_name

    def reset(self, **kwargs):
        # MineRL is old gym and doesn't accept gymnasium-style kwargs
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        for attempt in range(self.max_retries):
            try:
                return self.env.reset(**kwargs)
            except (TimeoutError, Exception) as e:
                logger.warning(
                    f"Reset failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                try:
                    self.env.close()
                except Exception:
                    pass
                time.sleep(5)
                self.env = create_environment(objective=self.env_name)
        raise RuntimeError("Environment failed to reset after max retries")

    def step(self, action):
        try:
            return self.env.step(action)
        except (TimeoutError, Exception) as e:
            logger.warning(f"Step failed, resetting environment: {e}")
            try:
                self.env.close()
            except Exception:
                pass
            time.sleep(5)
            self.env = create_environment(objective=self.env_name)
            obs = self.env.reset()  # no seed/options — MineRL is old gym
            return obs, 0.0, True, {"env_restarted": True}
