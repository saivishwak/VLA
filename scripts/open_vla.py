from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2
import gymnasium as gym
import numpy as np
from VLA import *
from VLA.tasks.cameras import Oracle
from VLA.utils import utils

task = tasks.names["align-box-corner"]()

record_cfg = {
    'save_video': False,
    'save_video_path': f'./videos/',
    'add_text': False,
    'fps': 20,
    'video_height': 640,
    'video_width': 720
}

env = Environment(
    None,
    disp=True,
    hz=480,
    task=task,
    record_cfg=record_cfg
)


observation, info = env.reset(seed=42)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    # [Optional] Requires `flash_attn`
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

prompt = "In: What action should the robot take to Pick the Brown block?\nOut:"

# # env.start_rec(f'test_rec')
max_tries = 25
num_tries = 0
while True:
    # Oracle camera is top down view of the table
    oracle_camera_config = Oracle.CONFIG[0]
    front_cam_config = env.agent_cams[0]
    left_cam_config = env.agent_cams[1]
    image, _, _ = env.render_camera(
        front_cam_config)
    color = np.array(image)
    img = Image.fromarray(color)
    # img.show()

    # Predict Action (7-DoF; un-normalize for BridgeV2)
    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(
        **inputs, unnorm_key="bridge_orig", do_sample=False)

    agent = task.oracle(env)
    p = agent.act(observation, info)

    # print("TEst", (np.array(action[:3]),
    #                np.array(utils.eulerXYZ_to_quatXYZW(action[3:-1]))))

    robot_action = {
        'pose0': (np.array(action[:3]),
                  np.array(utils.eulerXYZ_to_quatXYZW(action[3:-1]))),
        'pose1': p["pose1"]
    }

    observation, reward, done, info = env.step(robot_action)

    # print("Observation: ", observation)
    print("done", done)
    if done or (num_tries == max_tries):
        # observation, info = env.reset()
        break
    else:
        num_tries += 1
        # prompt += str(robot_action["pose0"])
        # prompt += "\nIn: You did not pick the Brown Block properly, What action you need to take to complete the task?\nOut:"
        # env.reset(seed=42)

# env.end_rec()
env.close()
