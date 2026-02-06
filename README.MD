# HERMES: Teaching a Robot to Walk

Hi! Welcome to **HERMES**.

This project is basically a gym for a digital robot. The goal is simple: we want to teach a humanoid robot how to stand up, walk around, and get back up if it falls down.

We use **Reinforcement Learning** (which is just a fancy way of saying "learning by trial and error") to teach it these skills. It tries to move, gets points for doing good things (like staying upright), and loses points for bad things (like falling over).

## What makes this cool?

Most robots just try to "move forward". HERMES is a bit smarter because it thinks in terms of **skills**:
1.  **Stand**: Just trying not to fall over.
2.  **Walk**: Moving forward while staying balanced.
3.  **Recover**: Figuring out how to get back up when it hits the ground.

It switches between these skills depending on what's happening. If it's walking and trips, it switches to "Recover" mode to get back up.

## The optional "Brain" (LLM)

For fun, we also added a feature where you can use a local AI (like Llama 3 running on your computer via Ollama) to act as a coach. It looks at the charts and stats from a training session and writes a report on how the robot can improve!

## How to run it

If you want to play around with it, here is the simple version:

1.  **Get the code**:
    ```bash
    git clone <repository-url>
    cd hermes
    ```

2.  **Install the needed tools**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the robot**:
    ```bash
    python -m rl.train
    ```
    This will start the training. You'll see the robot trying its best!

4.  **Project Phases**:
    - **Right now**: The robot is learning the basic skills (Stand, Walk, Recover).
    - **Coming soon**: We want to make the "Coach" AI smarter so it can help during the training, not just after.

## A quick look at the folders

- `skills/`: This is where the logic for "Stand", "Walk", and "Recover" lives.
- `rl/`: The main training code.
- `reports/`: Where the "report cards" for the training sessions go.
- `supervisor/`: The code for the AI Coach.

---
*That's it! It's a work in progress, but it's really fun to watch it learn.*
