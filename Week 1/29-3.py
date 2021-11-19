import numpy as np
from tqdm import tqdm


def random_walk(T, epsilon, p_left=.5):
    distance_walked = 0

    for _ in range(T):
        if np.random.random() > p_left:
            distance_walked += epsilon
        else:
            distance_walked -= epsilon

    return distance_walked


def main():
    T = 10000
    print(f"number of steps:",T)

    print("random walk...")
    for e in range(1,10):
        # For each epsilon (step size) sample 10000 random walks with T steps
        d = np.array([random_walk(T, e) for _ in tqdm(range(10000))])

        print(f"{e}:\tRMS: {np.sqrt(np.mean(d**2))}\tTheoretical: {np.sqrt(T)*e}")


if __name__ == '__main__':
    main()