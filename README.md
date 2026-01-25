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

## MineRL Setup
- MineRL is a powerful reinforcement learning framework for training agents. However builds will not work unless you
have `java` version `8` or `11` installed on your system.

### MACOS
1. First verify that you have the correct java version installed.
```bash
java --version
```
2. If you do not, please install `java 8` via `brew`
```bash
brew install temurin@8
```
3. Now ensure that you use `java 8`
```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```
4. Ensure that your python version is `3.10`, create a virtual environment using `make venv`
5. Install MineRL in the virtual environment `pip install git+https://github.com/minerllabs/minerl -v`
6. Follow this thread to patch the `MCP` folder
```bash
git clone https://github.com/minerllabs/minerl.git
sed -i .bak 's/3\.2\.1/3.3.1/' ./minerl/scripts/mcp_patch.diff
cd minerl && python setup.py
sed -i .bak s/'java -Xmx\$maxMem'/'java -Xmx\$maxMem -XstartOnFirstThread'/ ./minerl/MCP-Reborn/launchClient.sh
sed -i .bak /'GLFW.glfwSetWindowIcon(this.handle, buffer);'/d ./minerl/MCP-Reborn/src/main/java/net/minecraft/client/MainWindow.java
sed -i .bak '125,136s/^/\/\//' ./minerl/MCP-Reborn/src/main/java/net/minecraft/client/MainWindow.java
cd minerl/MCP-Reborn && ./gradlew clean build shadowJar
cd ../../../ && poetry add "git+https://github.com/minerllabs/minerl"
cp -rf minerl/minerl/MCP-Reborn/* .venv/lib/python3.10/site-packages/minerl/MCP-Reborn/
```
4. It is a [known issue](https://github.com/minerllabs/minerl/issues/659#issuecomment-1306635414) that running `MineRL`
   may take extra work on macOS. Follow this thread to ensure that everything is up and running smoothly.

# Running
```bash
make run
make interactor INTERACTIVE_PORT=6666 #defaults to 6666 but can configure it
```
