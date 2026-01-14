from mcpi.minecraft import Minecraft
import time

mc = Minecraft.create("localhost", 4711)

mc.postToChat("Hello from Python!")

x, y, z = mc.player.getPos()
mc.postToChat(f"Player at {x}, {y}, {z}")

mc.setBlock(x, y - 1, z, 41)  # Gold block under player

time.sleep(1)
mc.postToChat("Done!")

