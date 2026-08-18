# SuperPoint accuracy reference

`fp32-a65-tum-desk.npz` contains eight grayscale frames sampled across the bundled TUM RGB-D
`freiburg1_desk` video and the corresponding FP32 ONNX outputs decoded with the A65V1 profile.
The test compares a compiled MPK against these reference keypoints and descriptors; this is not a
calibration dataset.

- Video frame indices: `0, 80, 160, 240, 320, 400, 480, 560`
- Reference archive SHA-256: `90b914c5bacb3b453d8634c2429f380c27ae9916e709f449f65006576b15bbcf`
- Source model SHA-256: `81e38c7886f13b7448c6b844acf7a60bda8178e4a66df13d582fd729bbcd6b8d`
