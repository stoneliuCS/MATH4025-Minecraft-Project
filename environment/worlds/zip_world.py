#!/usr/bin/env python3
"""
Zip a Minecraft world folder for use with MineRL's LoadWorldAgentStart.

MineRL expects a specific zip structure:
    ./saves/<world_name>/level.dat
    ./saves/<world_name>/region/
    ./saves/<world_name>/...

Usage:
    python zip_world.py <world_folder> [output.zip]

Example:
    python zip_world.py simple
    python zip_world.py simple my_world.zip
"""

import argparse
import os
import zipfile
from pathlib import Path


def zip_world(world_folder: str, output_zip: str = None) -> str:
    """
    Create a MineRL-compatible zip from a Minecraft world folder.

    Args:
        world_folder: Path to the Minecraft world folder (containing level.dat)
        output_zip: Output zip file path. Defaults to <world_name>.zip

    Returns:
        Path to the created zip file
    """
    world_path = Path(world_folder).resolve()
    world_name = world_path.name

    if not world_path.exists():
        raise FileNotFoundError(f"World folder not found: {world_path}")

    if not (world_path / "level.dat").exists():
        raise ValueError(f"Not a valid Minecraft world: {world_path} (missing level.dat)")

    if output_zip is None:
        output_zip = world_path.parent / f"{world_name}.zip"
    else:
        output_zip = Path(output_zip).resolve()

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add directory entry first with ./ prefix
        # This ensures the first entry has the path structure MineRL expects
        dir_info = zipfile.ZipInfo(f"./saves/{world_name}/")
        zipf.writestr(dir_info, '')

        # Add all files with ./ prefix
        for root, dirs, files in os.walk(world_path):
            for file in sorted(files):
                filepath = Path(root) / file
                rel_path = filepath.relative_to(world_path)
                arcname = f"./saves/{world_name}/{rel_path}"

                info = zipfile.ZipInfo(arcname)
                with open(filepath, 'rb') as f:
                    zipf.writestr(info, f.read())

    print(f"Created: {output_zip}")
    return str(output_zip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zip a Minecraft world for MineRL LoadWorldAgentStart"
    )
    parser.add_argument("world_folder", help="Path to the Minecraft world folder")
    parser.add_argument("output", nargs="?", help="Output zip file (default: <world_name>.zip)")

    args = parser.parse_args()
    zip_world(args.world_folder, args.output)
