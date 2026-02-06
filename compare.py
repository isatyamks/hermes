import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def load_report(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_metrics(report):
    episodes = report["episodes"]
    n = len(episodes)

    episode_lengths = np.array([e["episode_length"] for e in episodes])
    energies = np.array([e["energy"] for e in episodes])
    falls = np.array([1 if e["fell"] else 0 for e in episodes])

    skill_counts = Counter(e["skill"] for e in episodes)
    skill_ratios = {k: v / n for k, v in skill_counts.items()}

    return {
        "mean_episode_length": episode_lengths.mean(),
        "std_episode_length": episode_lengths.std(),
        "fall_rate": falls.mean(),
        "mean_energy": energies.mean(),
        "std_energy": energies.std(),
        "skill_counts": skill_counts,
        "skill_ratios": skill_ratios,
        "num_episodes": n,
    }


def plot_episode_length(A, B, save_path="episode_length.png"):
    values = [A["mean_episode_length"], B["mean_episode_length"]]
    errors = [A["std_episode_length"], B["std_episode_length"]]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Run A", "Run B"], values, yerr=errors, capsize=6)

    plt.ylabel("Mean Episode Length")
    plt.title("Stability Improvement")

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v, f"{v:.1f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_fall_rate(A, B, save_path="fall_rate.png"):
    values = [A["fall_rate"], B["fall_rate"]]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Run A", "Run B"], values)

    plt.ylabel("Fall Rate")
    plt.title("Safety Comparison (Lower is Better)")
    plt.ylim(0, 1)

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v, f"{v:.2f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_energy(A, B, save_path="energy.png"):
    values = [A["mean_energy"], B["mean_energy"]]
    errors = [A["std_energy"], B["std_energy"]]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Run A", "Run B"], values, yerr=errors, capsize=6)

    plt.ylabel("Mean Energy per Episode")
    plt.title("Energy Efficiency (Lower is Better)")

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v, f"{v:.1f}",
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_skill_ratios(A, B, save_path="skill_ratios.png"):
    skills = sorted(set(A["skill_ratios"]) | set(B["skill_ratios"]))

    A_vals = [A["skill_ratios"].get(s, 0) for s in skills]
    B_vals = [B["skill_ratios"].get(s, 0) for s in skills]

    x = np.arange(len(skills))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, A_vals, width, label="Run A")
    plt.bar(x + width/2, B_vals, width, label="Run B")

    plt.xticks(x, skills)
    plt.ylabel("Fraction of Episodes")
    plt.title("Skill Usage Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    report_A_path = "reports/run_2025-12-19_15-15-00.json"
    report_B_path = "reports/run_2025-12-19_21-00-26.json"

    A = compute_metrics(load_report(report_A_path))
    B = compute_metrics(load_report(report_B_path))

    print("Run A:", A)
    print("Run B:", B)

    plot_episode_length(A, B)
    plot_fall_rate(A, B)
    plot_energy(A, B)
    plot_skill_ratios(A, B)

    print("Saved figures:")
    print("- episode_length.png")
    print("- fall_rate.png")
    print("- energy.png")
    print("- skill_ratios.png")
