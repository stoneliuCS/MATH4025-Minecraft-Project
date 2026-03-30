import numpy as np
from minerl.herobraine.hero.handlers.translation import (
    TranslationHandler,
    TranslationHandlerGroup,
)
from minerl.herobraine.hero import spaces


class ObservationFromRay(TranslationHandlerGroup):
    def to_string(self):
        return "ray"

    def __init__(self):
        super().__init__(handlers=[_RayObservation()])

    def xml_template(self) -> str:
        return "<ObservationFromRay/>"


class _RayObservation(TranslationHandler):
    def __init__(self):
        self.block_map = {
            "minecraft:oak_log": 1,
            "minecraft:birch_log": 2,
            "minecraft:spruce_log": 3,
        }
        super().__init__(
            spaces.Dict(
                {
                    "hit_type": spaces.Discrete(4),
                    "type": spaces.Dict(
                        {
                            "oak_log": spaces.Box(
                                low=0, high=1, shape=(), dtype=np.int32
                            ),
                            "birch_log": spaces.Box(
                                low=0, high=1, shape=(), dtype=np.int32
                            ),
                            "spruce_log": spaces.Box(
                                low=0, high=1, shape=(), dtype=np.int32
                            ),
                        }
                    ),
                    "distance": spaces.Box(
                        low=0.0, high=100.0, shape=(), dtype=np.float32
                    ),
                    "in_range": spaces.Discrete(2),
                    "x": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
                    "y": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
                    "z": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
                    "mine_block": spaces.Dict(
                        {
                            "oak_log": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                            "spruce_log": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                            "birch_log": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                            "oak_leaves": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                            "spruce_leaves": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                            "dirt": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                            "grass_block": spaces.Box(
                                low=0, high=1000, shape=(), dtype=np.int32
                            ),
                        }
                    ),
                }
            )
        )

        self.hit_type_map = {
            None: 0,
            "block": 1,
            "entity": 2,
            "item": 3,
        }

    def to_string(self):
        return "ray_data"

    def from_hero(self, obs):
        output = self.space.no_op()
        raw_mine = obs.get("mine_block", {})

        if isinstance(raw_mine, dict):
            for key in output["mine_block"].keys():
                output["mine_block"][key] = np.int32(raw_mine.get(key, 0))

        if "LineOfSight" not in obs:
            print("[Ray] No LineOfSight in obs")
            return output

        los = obs["LineOfSight"]
        print(f"[Ray] LineOfSight: type={los.get('type')} hitType={los.get('hitType')} inRange={los.get('inRange')} distance={los.get('distance', 0.0):.2f}")
        block_name = los.get("type", "")

        for key in output["type"].keys():
            output["type"][key] = np.int32(1 if key in block_name else 0)
        output["hit_type"] = self.hit_type_map.get(los.get("hitType"), 0)
        output["distance"] = np.float32(los.get("distance", 0.0))
        output["in_range"] = int(los.get("inRange", False))

        output["x"] = np.float32(los.get("x", 0.0))
        output["y"] = np.float32(los.get("y", 0.0))
        output["z"] = np.float32(los.get("z", 0.0))

        return output

    def from_universal(self, obs):
        output = self.space.no_op()

        try:
            los = obs["line_of_sight"]

            block_name = los.get("type", "")
            output["type"] = self.block_map.get(block_name, 0)
            output["hit_type"] = self.hit_type_map.get(los.get("hit_type"), 0)
            output["distance"] = np.float32(los.get("distance", 0.0))
            output["in_range"] = int(los.get("in_range", False))

            output["x"] = np.float32(los.get("x", 0.0))
            output["y"] = np.float32(los.get("y", 0.0))
            output["z"] = np.float32(los.get("z", 0.0))

        except KeyError:
            pass

        return output

