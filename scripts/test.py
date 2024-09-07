import gymnasium as gym
import numpy as np
from VLA import tasks, Environment

task = tasks.names["align-box-corner"]()

# env = gym.make('Environment/World-v0',
#                disp=True, hz=480, task=task, assets_root=None)

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
action = env.action_space.sample()
# Manual Pose value -(np.array([0.0, 0, 0]), np.array([0.0, 0, 0.01, 0.0]))
# print("action", action)

# action = {
#     'pose0': task.get_box_pose(),
#     'pose1': task.get_corner_pose()
# }
agent = task.oracle(env)
p = agent.act(observation, info)
# print("POSE", p)
action = {
    'pose0':    p["pose0"],
    'pose1': p["pose1"]
}

# env.start_rec(f'test_rec')

while True:
    observation, reward, done, info = env.step(action)
    # print("Observation: ", observation)
    print("done", done)
    if done:
        # observation, info = env.reset()
        break

# env.end_rec()
env.close()
