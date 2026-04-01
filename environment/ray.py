# import logging
# import numpy as np
# from minerl.herobraine.hero.handlers.translation import TranslationHandler
# from minerl.herobraine.hero import spaces


# class RayObservation(TranslationHandler):
#     """
#     Python handler for <ObservationFromRay>.
#     Produces a structured observation of the object under the cursor.
#     """

#     logger = logging.getLogger(__name__ + ".RayObservation")

#     def __init__(self):
#         super().__init__(spaces.Dict({
#             "hit_type": spaces.Discrete(4),  # none, block, entity, item
#             # "type": spaces.Text(32),         # block/entity/item name
#             "distance": spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32),
#             "in_range": spaces.Discrete(2),
#             "x": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
#             "y": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),

#             "z": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
            # "mine_block": spaces.Dict({
            #     "oak_log": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
            #     "spruce_log": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
            #     "birch_log": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
            #     "oak_leaves": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
            #     "dirt": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
            #     "grass_block": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
            # })
#         }))

#         self.hit_type_map = {
#             None: 0,
#             "block": 1,
#             "entity": 2,
#             "item": 3,
#         }

#     def to_string(self):
#         return "ray"

#     def xml_template(self) -> str:
#         return ""
#     # def add_to_mission_spec(self, mission_spec):
#     #     print("Adding RayObservation to mission spec")
#     #     mission_spec.observeRay(includeNBT=False)

#     def from_hero(self, obs):
#         # print("Processing hero observation for RayObservation")
#         """
#         Converts Malmo 'LineOfSight' JSON to structured observation.
#         """
#         # print("Received hero observation:", obs.keys())
#         mine = obs.get("mine_block", None)
#         # print("mine_block raw:", repr(obs.get("mine_block")))
#         # print("Raw observation:", obs.keys())
#         output = self.space.no_op()
#         # print("Initial output (no-op):", output)

#         raw_mine = obs.get("mine_block", {})

#         if isinstance(raw_mine, dict):
#             for key in output["mine_block"].keys():
#                 output["mine_block"][key] = np.int32(raw_mine.get(key, 0))
#                 # print(f"Processed mine_block {key}: {output['mine_block'][key]}")
        
#         if "LineOfSight" not in obs:
#             return output

#         los = obs["LineOfSight"]

#         hit_type_str = los.get("hitType")
#         output["hit_type"] = self.hit_type_map.get(hit_type_str, 0)
#         # output["mine_block"] = obs.get("mine_block")

#         output["type"] = los.get("type", "")

#         output["distance"] = np.float32(los.get("distance", 0.0))
#         output["in_range"] = int(los.get("inRange", False))

#         output["x"] = np.float32(los.get("x", 0.0))
#         output["y"] = np.float32(los.get("y", 0.0))
#         output["z"] = np.float32(los.get("z", 0.0))
#         print("RayObservation processed:", output)
#         return output

#     def from_universal(self, obs):

#         """
#         If universal observation format includes line_of_sight.
#         """
#         output = self.space.no_op()

#         try:
#             los = obs["line_of_sight"]

#             hit_type_str = los.get("hit_type")
#             output["hit_type"] = self.hit_type_map.get(hit_type_str, 0)

#             output["type"] = los.get("type", "")
#             output["distance"] = np.float32(los.get("distance", 0.0))
#             output["in_range"] = int(los.get("in_range", False))

#             output["x"] = np.float32(los.get("x", 0.0))
#             output["y"] = np.float32(los.get("y", 0.0))
#             output["z"] = np.float32(los.get("z", 0.0))

#         except KeyError:
#             self.logger.warning("Missing line_of_sight in universal observation.")

#         return output

import logging
import numpy as np
from minerl.herobraine.hero.handlers.translation import TranslationHandler, TranslationHandlerGroup
from minerl.herobraine.hero import spaces


# 🔥 GROUP HANDLER (this emits the XML)
class ObservationFromRay(TranslationHandlerGroup):

    def to_string(self):
        return "ray"

    def __init__(self):
        # print("[RayObservation] INIT CALLED")  # ✅ confirms handler constructed
        super().__init__(handlers=[
            _RayObservation()
        ])

    def xml_template(self) -> str:
        # print("[RayObservation] XML TEMPLATE CALLED")
        return "<ObservationFromRay/>"


# 🔍 ACTUAL PARSER
class _RayObservation(TranslationHandler):

    logger = logging.getLogger(__name__ + "._RayObservation")

    def __init__(self):
        # print("[_RayObservation] INIT CALLED")
        self.block_map = {
            "minecraft:oak_log": 1,
            "minecraft:birch_log": 2,
            "minecraft:spruce_log": 3,
        }
        super().__init__(spaces.Dict({
            "hit_type": spaces.Discrete(4),
            "type": spaces.Dict({
                "oak_log": spaces.Box(low=0, high=1, shape=(), dtype=np.int32),
                "birch_log": spaces.Box(low=0, high=1, shape=(), dtype=np.int32),
                "spruce_log": spaces.Box(low=0, high=1, shape=(), dtype=np.int32),
            }),

            "distance": spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32),
            "in_range": spaces.Discrete(2),
            "x": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
            "y": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
            "z": spaces.Box(low=-1e6, high=1e6, shape=(), dtype=np.float32),
            "mine_block": spaces.Dict({
        "oak_log": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
        "spruce_log": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
        "birch_log": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
        "oak_leaves": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
        "spruce_leaves": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
        "dirt": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
        "grass_block": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
    }),
        }))

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
        # print("FULL OBS:", obs)
        # print(f" hi{obs.keys()}")
        raw_mine = obs.get("mine_block", {})

        if isinstance(raw_mine, dict):
            for key in output["mine_block"].keys():
                output["mine_block"][key] = np.int32(raw_mine.get(key, 0))

        if "LineOfSight" not in obs:
            # print("[_RayObservation] ❌ NO LineOfSight FOUND")
            return output
        # print("[_RayObservation] ✅  LineOfSight FOUND")

        los = obs["LineOfSight"]
        # print("[_RayObservation] LineOfSight data:", los)
        # output["type"] = los.get("type", "")
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