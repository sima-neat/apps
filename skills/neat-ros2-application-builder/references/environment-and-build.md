# Environment and build

## Identify the actual installation

Distinguish the development workstation, Neat Development Environment and Modalix
runtime. Confirm architecture, Neat release, ROS distribution, Python executable
and ABI, and available build tools. Use `neat --json` when the installed CLI
supports it. Inspect installed headers/bindings and matching SDK documentation;
an unrelated Core checkout or online example cannot establish runtime support.

Core-bundled SDK images can provide `/neat-resources/core-src` and
`/neat-resources/apps-src`. Discover these rather than requiring them. Relevant
Core documentation is under `docs/develop-apps/` and `docs/reference/`; public
headers are in the target sysroot or target installation. Search for the specific
API being changed. Use the [public Apps repository](https://github.com/sima-neat/apps)
as an example source when needed, checking release compatibility.

Use the user's existing working environment. The supported ROS bundle command
for the v3.0.0 installation workflow is:

```bash
sima-cli neat install ros2@v3.0.0
```

This is a release-specific example, not an instruction to upgrade every board.
Check the selected SDK's installation guidance and `sima-cli neat install --help`
when targeting another release. ROS, Neat and Model Zoo versions are distinct;
do not derive all three from one version string.

ROS runtime availability does not prove colcon is installed. After sourcing the
installation, check `command -v ros2` and `command -v colcon`, then import `rclpy`
and `pyneat` using the interpreter that will run the node. If colcon is missing,
check release packaging first. Use documented build-tool packages or an isolated,
ABI-compatible tools environment when required; do not repair it by replacing
system Python, ROS dependencies or the Neat environment indiscriminately.

## Build where the package supports it

The following source the conventional ROS installation and an example Neat
virtual environment. Confirm these paths exist; substitute the actual environment.
From the standalone package directory:

```bash
source /usr/local/ros2/setup.bash
source "$HOME/pyneat/bin/activate"
python -c 'import sys, rclpy, pyneat; print(sys.executable)'
command -v colcon
colcon build
source install/setup.bash
```

Use a fresh terminal when switching independent packages with different overlays.
In a larger workspace, select the intended package and its dependencies using the
workspace's normal colcon workflow. Do not require building all examples to run
one standalone package. `--packages-select` does not build missing dependencies;
use the appropriate dependency-aware selection when needed.

For `ament_cmake`, inspect the installed exported Neat CMake package and targets
before linking; do not invent library names. For cross-compilation, use the SDK's
supported toolchain, target ROS libraries and target ABI. A native x86 build is
not evidence that a binary will run on Modalix. Python nodes still depend on
compatible native ROS and Neat modules.

A ROS source package and a native Apps runtime bundle are different deliverables.
Do not assume a runtime-only bundle contains `package.xml`, launch files or CMake
sources. Follow the selected distribution's actual supported workflow. Avoid
adding ROS requirements to a native-only repository build just to compile examples.

## Verify the installed result

- Install executables, imported helpers, launch files, labels and configs. Test
  imports from the installed package, not just a source directory on `PYTHONPATH`.
- Check `ros2 pkg prefix <package>` and the selected config path before a run.
  Rebuild after code/launch changes. YAML edits require a rebuild only if running
  a stale installed copy rather than the edited source YAML.
- Test launch from the documented working directory. Resolve packaged resources
  through the package share directory; document how user model/input paths resolve.
  A path beginning `examples/...` is usually wrong when the working directory is
  already inside an example.
- Run only the relevant existing tests. If the package registers colcon tests,
  use `colcon test` and `colcon test-result --verbose` for that package.

## Target work and delivery

Use authorized device access and a writable test directory with enough storage.
Check competing workloads before measuring. A separate `ROS_DOMAIN_ID` isolates
ROS discovery, not accelerator, memory, CPU or Insight ports.

Copy the current files, including intended uncommitted changes; omit local
build/install/log trees. Preserve executable permissions and relevant relative
resources. Verify hashes on the target and record the snapshot used for testing.
Do not overwrite a user's tuned config when deploying a new test copy.

Keep models outside disposable build folders. Use the selected model's official
acquisition/preparation instructions, matching target and variant; a downloaded
ZIP is not necessarily the model archive Neat loads. Do not download all models
for a focused transport test. Keep credentials, board addresses and model paths
out of reusable source and skill content.
