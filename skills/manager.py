class SkillManager:


    def __init__(self, skills, initial_skill):
        self.skills = {s.name: s for s in skills}
        self.active_skill = self.skills[initial_skill]

    def set_skill(self, skill_name):
        if skill_name not in self.skills:
            raise ValueError(f"Unknown skill: {skill_name}")
        self.active_skill = self.skills[skill_name]

    def compute_reward(self, obs, action, env_reward):
        return self.active_skill.reward(obs, action, env_reward)

    def should_terminate(self, obs):
        return self.active_skill.termination(obs)

    @property
    def name(self):
        return self.active_skill.name
