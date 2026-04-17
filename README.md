# VLM-based Robotic Manipulation

This project implements a VLM-based action server and SAPIEN simulation client for articulated object manipulation (drawers and cabinet doors).

## 1. Environment Configuration

### Conda Environment
Create and activate the environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate owlsam
```

### MobileSAM Installation
Install the MobileSAM library from the [official GitHub repository](https://github.com/ChaoningZhang/MobileSAM):
```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```
**Note**: The pre-trained weights (`mobile_sam.pt`) are already provided in the `weights/` directory.

## 2. Execution Workflow

To run the full pipeline, open three separate terminal windows and run the following commands in order:

### Terminal 1: Detection Backend
Starts the OWLv2 + MobileSAM detection server (FastAPI).
```bash
python det_pipeline.py
```

### Terminal 2: SAPIEN Simulation Client
Loads the robot and the cabinet model (e.g., PartNet-Mobility 40147).
```bash
python client_sapien_40147.py
```

### Terminal 3: Action Server
Starts the planner, 3D visualizer, and manipulation state machine.
```bash
python action_server.py
```

## 3. Operation

1. Ensure all three components (Detection, Client, Server) are running and connected.
2. Focus the **SAPIEN simulator window**.
3. Press **`Space`** to trigger the Action Server to begin the detection and manipulation sequence.

## 4. Important Notes

*   **VLM Connectivity**: The system currently uses a heuristic stub for planning. It does not have a live connection to a real VLM API yet.
*   **Action Planning**: The task plan (target object, motion type) is defined in the `call_vlm` function within `action_server.py`. To manipulate different parts or switch between `Translation` and `Rotation`, you must manually update the return JSON in that stub.
*   **Geometric Estimation**: The server uses 3D spherical sampling combined with SAM masks for precise normal and hinge estimation. You can monitor this in the 3D visualizer (Red = normal points, Blue = hinge points, Pink Star = rotation axis).

---
