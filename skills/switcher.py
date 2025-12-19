class SkillSwitcher:

    def __init__(self, skill_manager):
        self.skill_manager = skill_manager

    def decide(self, episode_summary):
        current = self.skill_manager.name

        mean_height = episode_summary["mean_height"]
        fell = episode_summary["fell"]
        length = episode_summary["episode_length"]

        
        if current == "stand":
            if mean_height > 1.2 and length > 50:
                return "walk"

        
        if current == "walk":
            if fell:
                return "recover"

        
        if current == "recover":
            if not fell and mean_height > 1.0:
                return "stand"

        return current
