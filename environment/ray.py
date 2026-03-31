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
            "minecraft:jungle_log": 4,
            "minecraft:acacia_log": 5,
            "minecraft:dark_oak_log": 6,
            "minecraft:stripped_oak_log": 7,
            "minecraft:stripped_spruce_log": 8,
            "minecraft:stripped_birch_log": 9,
            "minecraft:stripped_jungle_log": 10,
            "minecraft:stripped_acacia_log": 11,
            "minecraft:stripped_dark_oak_log": 12,
        }

        def _log_flag():
            return spaces.Box(low=0, high=1, shape=(), dtype=np.int32)

        def _count():
            return spaces.Box(low=0, high=1000, shape=(), dtype=np.int32)

        super().__init__(
            spaces.Dict(
                {
                    "hit_type": spaces.Discrete(4),
                    "type": spaces.Dict(
                        {
                            "oak_log": _log_flag(),
                            "birch_log": _log_flag(),
                            "spruce_log": _log_flag(),
                            "jungle_log": _log_flag(),
                            "acacia_log": _log_flag(),
                            "dark_oak_log": _log_flag(),
                            "stripped_oak_log": _log_flag(),
                            "stripped_spruce_log": _log_flag(),
                            "stripped_birch_log": _log_flag(),
                            "stripped_jungle_log": _log_flag(),
                            "stripped_acacia_log": _log_flag(),
                            "stripped_dark_oak_log": _log_flag(),
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
                            "oak_log": _count(),
                            "spruce_log": _count(),
                            "birch_log": _count(),
                            "jungle_log": _count(),
                            "acacia_log": _count(),
                            "dark_oak_log": _count(),
                            "stripped_oak_log": _count(),
                            "stripped_spruce_log": _count(),
                            "stripped_birch_log": _count(),
                            "stripped_jungle_log": _count(),
                            "stripped_acacia_log": _count(),
                            "stripped_dark_oak_log": _count(),
                            "oak_leaves": _count(),
                            "spruce_leaves": _count(),
                            "birch_leaves": _count(),
                            "jungle_leaves": _count(),
                            "acacia_leaves": _count(),
                            "dark_oak_leaves": _count(),
                            "dirt": _count(),
                            "grass_block": _count(),
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

