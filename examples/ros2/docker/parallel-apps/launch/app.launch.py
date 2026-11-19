"""Run the configured container composition until Ctrl+C."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    common = Path(get_package_share_directory('neat_parallel_apps')) / 'common'
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(common / 'config.yaml')),
        ExecuteProcess(cmd=['bash', str(common / 'ros-parallel-apps.sh'), 'run',
                            LaunchConfiguration('config')], output='screen'),
    ])
