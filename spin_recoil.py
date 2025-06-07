import mujoco
import numpy as np
from mujoco import viewer
import time
import os

def run_simulation(xml_path):
    # Load model from XML file
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize viewer
    viewer_handle = viewer.launch_passive(model, data)

    # Apply keyframe if exists
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)  # Apply first keyframe

    # Simulation loop
    try:
        while viewer_handle.is_running():
            step_start = time.time()
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer_handle.sync()
            
            # Pause to maintain real-time simulation
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        viewer_handle.close()

if __name__ == "__main__":
    xml_file = "spin_recoil.xml"
    
    # Verify file exists
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
    
    print(f"Running simulation from {xml_file}...")
    run_simulation(xml_file)