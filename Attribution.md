# Attribution

This document identifies all sources of contribution to this project, including human authorship and AI-assisted content generation. All code, documentation, and written components are accurately attributed in accordance with course policy and standard academic integrity practices.

---

## Human Contributor

**Paul Dilly**  
Role: Primary author and developer  
Contributions:
- Designed and implemented the field-goal machine learning project concept
- Wrote all reinforcement learning environment logic, DQN agent implementation, and reward model design
- Implemented data ingestion, feature engineering, and supervised learning components (`data.py`, `model.py`)
- Authored all experimental decisions, code integration, debugging, and parameter tuning
- Wrote structural project organization and final analysis included in project deliverables
- Verified correctness of AI-generated text and incorporated or modified it as needed

### Code Components Fully Written by the Human Author
- The custom Gymnasium environment (`FieldGoalEnv`) including physics calculations, kick trajectory simulation, wind modeling, temperature scaling, slip coefficient mechanics, reward shaping logic, and termination conditions
- The complete DQN training loop and agent logic (`FieldGoalAgent`, replay buffer, epsilon decay, target network updates)
- The supervised learning model implementation, feature engineering, and data preprocessing pipeline (`data.py`, `model.py`)
- All experimental configuration decisions and hyperparameter choices
- All evaluation protocol design and integration into the notebook workflows

---

## AI-Assisted Contributions

This project used AI (ChatGPT by OpenAI) as a writing and coding assistant. The following components include AI-generated or AI-assisted content. All AI-generated material was reviewed, edited, or integrated by the human author.

### Documentation Generated or Substantially Assisted by AI
- `SETUP.md`  
- `README.md`  
- `ATTRIBUTION.md`  

### Code Components Partially Generated or Assisted by AI
- Plotting and visualization functions used in the reinforcement learning evaluation workflow
- Helper functions for saving plots and formatting output
- Some boilerplate sections for initializing GPU devices and utility print statements
- Internal text descriptions for the supervised and reinforcement learning methodology

---

## External Libraries and Tools Used

This project uses the following external libraries:

- Python 3.9+  
- PyTorch (for neural networks and DQN implementation)  
- Gymnasium with Box2D (for RL environment compatibility)  
- NumPy, Pandas, Matplotlib (data handling and visualization)  
- scikit-learn (supervised model utilities and logistic regression feature analysis)  
- JupyterLab / IPykernel (interactive experimentation)

No external code outside these officially provided libraries was copied into the project.

---

## Data Sources

Supervised learning uses NFL play-by-play data processed locally via `data.py`.  
No copyrighted datasets or proprietary material were copied from online sources.  
All reinforcement learning data is generated entirely through the custom simulation environment developed for this project.

---

## Summary

This project is a collaborative effort between human authorship and AI-assisted components.  
AI tools were used as a utility for:
- Documentation scaffolding
- Visualization code generation
- Formatting assistance

All core logic, modeling decisions, environment engineering, and machine learning implementation were created by **Paul Dilly** and constitute the primary intellectual contribution of the project.
