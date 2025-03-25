from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool
import yaml

# External search tool (useful for gathering latest updates)
search_tool = SerperDevTool()

# Output schema for Dream11 team
class Dream11TeamPrediction(BaseModel):
    match: str
    team1: str
    team2: str
    dream11_team: list[str] = Field(..., description="List of selected players for Dream11")
    captain: str = Field(..., description="Captain of Dream11 team")
    vice_captain: str = Field(..., description="Vice-Captain of Dream11 team")
    key_factors: list[str] = Field(..., description="Key factors influencing team selection")
    confidence_level: float = Field(..., description="Confidence in the team selection")

@CrewBase
class Dream11Predictor:
    """Dream11 IPL Predictor Crew"""

    agents_config = "config/agents.yaml"
    tasks_config="config/tasks.yaml"

    @before_kickoff
    def before_kickoff_function(self, inputs):
        team1 = input("Enter first IPL team: ")
        team2 = input("Enter second IPL team: ")
        match_date = input("Enter match date (DD-MM-YYYY): ")

        inputs['team1'] = team1
        inputs['team2'] = team2
        inputs['match'] = f"{team1} vs {team2} on {match_date}"
        print(f"\nAnalyzing Dream11 team for: {inputs['match']}")
        return inputs

    @after_kickoff
    def after_kickoff_function(self, result):
        try:
            print("\nDream11 Team Recommendation:")
            print(f"Match: {result.match}")
            print(f"Team: {', '.join(result.dream11_team)}")
            print(f"Captain: {result.captain}")
            print(f"Vice-Captain: {result.vice_captain}")
            print(f"Key Factors: {', '.join(result.key_factors)}")
            print(f"Confidence Level: {result.confidence_level * 100:.2f}%")
        except Exception as e:
            print(f"Unable to process Dream11 team results: {e}")
        return result

    @agent
    def player_stats_analyst(self) -> Agent:
        return Agent(config=self.agents_config['player_stats_analyst'], verbose=True)

    @agent
    def pitch_weather_expert(self) -> Agent:
        return Agent(config=self.agents_config['pitch_weather_expert'], verbose=True)

    @agent
    def team_form_researcher(self) -> Agent:
        return Agent(config=self.agents_config['team_form_researcher'], verbose=True)

    @agent
    def fantasy_strategy_guru(self) -> Agent:
        return Agent(config=self.agents_config['fantasy_strategy_guru'], verbose=True)

    @agent
    def dream11_selector(self) -> Agent:
        return Agent(config=self.agents_config['dream11_selector'], verbose=True)

    @task
    def player_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['player_analysis_task'])

    @task
    def pitch_weather_task(self) -> Task:
        return Task(config=self.tasks_config['pitch_weather_task'], tools=[search_tool])

    @task
    def team_form_task(self) -> Task:
        return Task(config=self.tasks_config['team_form_task'], tools=[search_tool])

    @task
    def fantasy_strategy_task(self) -> Task:
        return Task(config=self.tasks_config['fantasy_strategy_task'], tools=[search_tool])

    @task
    def dream11_team_selection_task(self) -> Task:
        return Task(config=self.tasks_config['dream11_team_selection_task'], output_json=Dream11TeamPrediction)

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.player_stats_analyst(),
                self.pitch_weather_expert(),
                self.team_form_researcher(),
                self.fantasy_strategy_guru(),
                self.dream11_selector()
            ],
            tasks=[
                self.player_analysis_task(),
                self.pitch_weather_task(),
                self.team_form_task(),
                self.fantasy_strategy_task(),
                self.dream11_team_selection_task()
            ],
            process=Process.sequential,
            verbose=True
        )

# Run function
def run():
    dream11_instance = Dream11Predictor()
    dream11_instance.crew().kickoff()
