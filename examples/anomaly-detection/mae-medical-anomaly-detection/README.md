# mae-medical-anomaly-detection

## Metadata
| Field | Value |
| --- | --- |
| Category | anomaly-detection |
| Difficulty | Intermediate |
| Tags | anomaly-detection, medical |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | mae-medical-anomaly-detection |
| Model | TODO |

## Concept
TODO: describe what this example demonstrates, e.g. using a MAE-style model for medical image anomaly detection.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>
cmake -S examples/anomaly-detection/mae-medical-anomaly-detection/cpp -B build/mae-medical-anomaly-detection
cmake --build build/mae-medical-anomaly-detection -j
```

## Run
### C++
```bash
./build/examples/anomaly-detection/mae-medical-anomaly-detection/mae-medical-anomaly-detection
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/anomaly-detection/mae-medical-anomaly-detection/python/requirements.txt
python3 examples/anomaly-detection/mae-medical-anomaly-detection/python/main.py
```

## Source Files
- C++: `cpp/main.cpp`
- C++ tests: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`
- Python: `python/main.py`
- Python tests: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- Shared assets: `common/`


