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
