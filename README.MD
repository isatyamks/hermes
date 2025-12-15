# HERMES Visualization System

Comprehensive visualization system for analyzing RL training, reports, LLM analysis, and evaluation results.

## Overview

The visualization system automatically generates detailed plots and graphs after training, providing in-depth analysis of:
- **Training Process**: Reward progression, skill performance, energy consumption, etc.
- **Report Analysis**: Aggregated statistics, skill comparisons, failure modes
- **LLM Analysis**: Confidence scores, suggested actions, parameter changes
- **Evaluation Results**: Performance metrics, trajectories, success rates

## Quick Start

### Automatic Generation

Visualizations are automatically generated after training:

```bash
python -m rl.train
```

This will create visualizations in `visualizations/outputs/training/` and `visualizations/outputs/reports/`.

### Manual Generation

Generate visualizations from existing data:

```bash
# From training report
python visualize.py --report supervisor/reports/run_YYYY-MM-DD_HH-MM-SS.json

# From LLM analysis
python visualize.py --analysis supervisor/analyses/analysis_YYYY-MM-DD_HH-MM-SS.json

# Both
python visualize.py --report <report_path> --analysis <analysis_path>
```

### After LLM Analysis

The `tr.py` script automatically generates visualizations:

```bash
python tr.py
```

## Visualization Types

### Training Visualizations (13 plots)

Generated automatically during training:

1. **Reward Progression** - Reward over episodes with rolling mean
2. **Episode Length** - Episode length progression
3. **Energy Trend** - Energy consumption over time
4. **Height Progression** - Mean height with fall threshold
5. **Fall Rate** - Rolling fall rate over episodes
6. **Skill Usage** - Pie and bar charts of skill distribution
7. **Skill Performance** - Multi-metric comparison by skill
8. **Skill Timeline** - Skill switching over episodes
9. **Reward Distribution** - Histogram of reward values
10. **Height Distribution** - Histogram of height values
11. **Energy vs Reward** - Scatter plot colored by skill
12. **Skill Transitions** - Heatmap of skill transition probabilities
13. **Multi-Metric Overview** - Combined view of key metrics

### Report Visualizations (9 plots)

Generated from training report JSON:

1. **Overall Dashboard** - Key statistics overview
2. **Skill Comparison** - Performance metrics by skill
3. **Failure Modes** - Fall type distribution
4. **Skill Efficiency** - Reward per episode by skill
5. **Radar Chart** - Multi-dimensional skill comparison
6. **Episode Counts** - Episode distribution by skill
7. **Fall Rate by Skill** - Fall rate comparison
8. **Recovery Analysis** - Recovery skill effectiveness
9. **Comprehensive Summary** - Full dashboard

### LLM Analysis Visualizations (5 plots)

Generated from LLM analysis JSON:

1. **Confidence Gauge** - Polar gauge showing confidence score
2. **Suggested Actions** - Visualization of improvement suggestions
3. **Parameter Changes** - Before/after comparisons
4. **Action Types** - Distribution of action types
5. **Analysis Summary** - Comprehensive overview

### Evaluation Visualizations (6 plots)

Generated from evaluation results:

1. **Performance Summary** - Key evaluation metrics
2. **Trajectories** - Reward, height, length, energy over episodes
3. **Success Rate** - Success vs failure breakdown
4. **Reward Distribution** - Histogram of evaluation rewards
5. **Length Distribution** - Histogram of episode lengths
6. **Evaluation Dashboard** - Comprehensive evaluation view

## Output Structure

```
visualizations/outputs/
├── training/          # Training-time visualizations
├── reports/           # Report analysis visualizations
├── llm/               # LLM analysis visualizations
├── evaluation/        # Evaluation visualizations
└── dashboards/        # Dashboard summaries
```

## Usage Examples

### Programmatic Usage

```python
from visualizations.training_analyzer import TrainingAnalyzer
from visualizations.report_analyzer import ReportAnalyzer
from visualizations.dashboard import DashboardGenerator

# Training visualizations
analyzer = TrainingAnalyzer()
plot_paths = analyzer.analyze(episode_summaries)

# Report visualizations
report_analyzer = ReportAnalyzer()
plot_paths = report_analyzer.analyze("path/to/report.json")

# Full dashboard
dashboard = DashboardGenerator()
dashboard.generate_full_dashboard(
    episode_summaries=episodes,
    report_path="path/to/report.json",
    analysis_path="path/to/analysis.json"
)
```

## Customization

### Output Directory

```python
analyzer = TrainingAnalyzer(output_dir="custom/path")
```

### Disable Plot Saving

```python
plot_paths = analyzer.analyze(episodes, save_plots=False)
```

### Custom Colors

Modify `get_skill_colors()` in `visualizations/utils.py` to customize skill colors.

## File Formats

All plots are saved as:
- **PNG** (default, 300 DPI)
- Can be extended to PDF, SVG via `save_figure()` formats parameter

## Integration

The visualization system is integrated into:

1. **Training Callback** (`rl/callbacks.py`): Auto-generates training plots
2. **Training Script** (`rl/train.py`): Generates report visualizations
3. **LLM Analysis** (`tr.py`): Generates LLM analysis visualizations
4. **Standalone Script** (`visualize.py`): Manual generation tool

## Requirements

Additional dependencies (already in `requirements.txt`):
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualizations

## Notes

- Visualizations use non-interactive backend (`Agg`) for server compatibility
- All plots include proper labels, legends, and grid lines
- Color schemes are consistent across all visualizations
- Plots are optimized for publication (300 DPI, tight layout)



