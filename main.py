import flappy_bird_gymnasium
import gymnasium
import random
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()
    action = 1
    # action = random.randint(0,1)

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    print(obs)
    # optuna
    # Checking if the player is still alive
    if terminated:
        break

env.close()