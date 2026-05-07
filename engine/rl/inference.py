from datacenter import DataCenterEnv
import numpy as np


def run_inference(model, input_data):
    env = DataCenterEnv(grid_size=50, rack_num=input_data["rack_num"])

    obs, _ = env.reset(
        options={
            "obstacle": np.array(input_data["obstacle"]),
            "cooling_pos": np.array(input_data["cooling_pos"]),
            "rack_num": input_data["rack_num"],
        }
    )

    done = False

    while not done:
        mask = env.get_action_mask()

        action, _ = model.predict(obs, action_masks=mask, deterministic=True)

        obs, _, done, _, _ = env.step(action)

    racks = np.argwhere(env.rack_map == 1)

    result = []
    for x, y in racks:
        result.append({"x": int(x), "y": int(y), "dir": int(env.rack_dir[x, y])})

    return result
