# Custom Reinforcement Learning environment for Edwin Lander

This repository contains the source code of Edwin Lander for training any Reinforcement Learning algorithm on it to make landing.

## Demo
https://user-images.githubusercontent.com/74960567/223930760-b2d111ca-074d-4322-80d7-29e53d6aaae0.mp4

## Usage:
1. Start a local server and host the files present in `lander` directory and get the hosted URL. For simplicity you can use `Go-Live` feature of VSCode
2. Replace `self.driver.get("<URL>")` with the hosted URL inside `main.py`
3. Pick the algorithm you want to use. (It should support BOX type action spaces)
4. run the `main.py`
