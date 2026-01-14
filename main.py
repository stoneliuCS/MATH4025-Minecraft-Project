from mcpi import minecraft

CONNECTION_STRING = "0.0.0.0"
CONNECTION_PORT = 4711


def connect_to_minecraft_server(address: str, port: int) -> minecraft.Minecraft:
    connection = minecraft.Connection(address, port)
    mc = minecraft.Minecraft(connection)
    return mc


if __name__ == "__main__":
    mc = connect_to_minecraft_server(CONNECTION_STRING, CONNECTION_PORT)
