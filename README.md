# Development Environment Setup
1. Download the `server.jar` file [here](https://getbukkit.org/download/spigot)
    - run the following command to start up the server (headless)
    ```bash
    java -Xmx1024M -Xms1024M -jar server.jar nogui
    ```
    - You may receive a `eula` error, go to the created `eula.txt` file and switch the value of `eula=false` to
    `eula=true`
    - Or you may wish to start the server with the gui
    ```bash
    java -Xmx1024M -Xms1024M -jar server.jar
    ```
2. 
    ```bash
    git clone https://github.com/zhuowei/RaspberryJuice
    cd RaspberryJuice
    mvn package
    ```
    - Make sure you have `maven` on your system before running. If you get an error regarding the source and target
    versions change the version to `1.8` instead of `1.7` in the `pom.xml` file. The jar file should exist
    - Ensure that you also put the jar file in the `/plugins` folder.
    ```bash
    RaspberryJuice/target/raspberryjuice-1.12.1.jar
    ```
```
#Minecraft server properties
#Wed Jan 14 15:21:01 EST 2026
accepts-transfers=false
allow-flight=false
broadcast-console-to-ops=true
broadcast-rcon-to-ops=true
bug-report-link=
debug=false
difficulty=peaceful
enable-code-of-conduct=false
enable-jmx-monitoring=false
enable-query=false
enable-rcon=false
enable-status=true
enforce-secure-profile=true
enforce-whitelist=false
entity-broadcast-range-percentage=100
force-gamemode=false
function-permission-level=2
gamemode=creative
generate-structures=true
generator-settings={"biome"\:"minecraft\:the_void","layers"\:[],"structures"\:{}}
hardcore=false
hide-online-players=false
initial-disabled-packs=
initial-enabled-packs=vanilla
level-name=world
level-seed=
level-type=minecraft\:flat
log-ips=true
management-server-allowed-origins=
management-server-enabled=false
management-server-host=localhost
management-server-port=0
management-server-secret=vaKTnMoWOMTXqXJUqynx8tqERPiU2v1ViNLIQ1s1
management-server-tls-enabled=true
management-server-tls-keystore=
management-server-tls-keystore-password=
max-chained-neighbor-updates=1000000
max-players=20
max-tick-time=60000
max-world-size=29999984
motd=A Minecraft Server
network-compression-threshold=256
online-mode=true
op-permission-level=4
pause-when-empty-seconds=60
player-idle-timeout=0
prevent-proxy-connections=false
query.port=25565
rate-limit=0
rcon.password=
rcon.port=25575
region-file-compression=deflate
require-resource-pack=false
resource-pack=
resource-pack-id=
resource-pack-prompt=
resource-pack-sha1=
server-ip=
server-port=25565
simulation-distance=10
spawn-protection=16
status-heartbeat-interval=0
sync-chunk-writes=true
text-filtering-config=
text-filtering-version=0
use-native-transport=true
view-distance=10
white-list=false
```
