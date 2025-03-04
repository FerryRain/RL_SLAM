# RL_SLAM

#### Paper: Autonomous localized path planning algorithm for UAVs based on TD3 strategy

January 20, 2023 AM

	Modify single training consortium to perform five

	Modify learning rate

	Modify to yolo network normalization, remove last two dimensions of yolo

January 20, 2023 21:28 PM

	Modify network width, batch_size, lr for testing

	Remove yaw dimension (initially thought to be useless, and increase number of states)

January 21, 2023 at 09:06:36

    Modify explore random_value, minimum_value, decay_steps

Optimization goals

    1. tweak noise values to find the right noise (compare test results)
    2. increase the number of steps per round, reward value set self.id penalty, encourage to reach the target point with the shortest number of steps (compare convergence speed)
    3. reward value add reward term, when pix_distance is less than a certain threshold make it get a reward for every amount of proximity (compare convergence speed)

Translated with DeepL.com (free version)