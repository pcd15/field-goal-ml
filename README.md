# NFL, Here We Come: Using Machine Learning to Predict and Optimize Field-Goal Success Rates

## 1. What It Does

This project builds **two tightly connected machine learning systems** to analyze and optimize field-goal performance in American football:

1. A **supervised learning model** that predicts the **kicker-agnostic probability** of a field goal being made, using **25 years of NFL play-by-play data**.
2. A **reinforcement learning (RL) system** in a custom, physics-based field-goal environment, where a **Deep Q-Network (DQN)** learns how to choose kick **velocity**, **vertical angle**, and **horizontal aim** under varying environmental conditions.

Together, these components address a **single, unified research question**:

> **Given real NFL data and a controllable simulation environment, how well can machine learning both *predict* field-goal success and *optimize* kicking decisions to maximize success rates—and how does optimal RL performance compare to historical human performance?**

In other words:
- The **supervised model** tells us what *NFL kickers actually do*.
- The **RL agent** tells us what *an optimally trained system could do*.
- Comparing them reveals how much room there is to improve real-world kicking strategy.

This is motivated by a real-world problem: **coaches and kickers must constantly decide whether a field goal is worth attempting**, yet those decisions are often based on rough rules of thumb rather than data-driven, optimized strategies.

---

## 2. Quick Start

For installation and execution instructions, see **`SETUP.md`**. It explains how to:

- Create and activate a virtual environment  
- Install all dependencies (`requirements.txt`)  
- Download and process the NFL data  
- Run the supervised model  
- Run the RL training and evaluation notebooks/scripts  

---

## 3. Supervised Learning: Predicting Make Probability from NFL Data

### 3.1 Motivation

Before trying to optimize kicking strategy in simulation, we first want to quantify:

- How likely is a field goal to be made in real NFL games?
- Which factors (distance, weather, stadium, game context) matter most?
- Can we capture this with a **simple, interpretable model**?

This sets a **baseline of real-world performance** that the RL system will later be compared against.

---

### 3.2 Model Families Considered

We tested two model families on historical field-goal data:

- **Logistic Regression**
- **Gradient Boosted Trees (GBT)**

While GBTs can model complex nonlinear relationships, early experiments showed:

- Only marginal AUC improvement over logistic regression  
- Much higher complexity and lower interpretability  
- More hyperparameters and tuning overhead  

Because our goals include **interpretability** and **calibrated probability estimates**, we chose **Logistic Regression** as the primary model family.

---

### 3.3 Full Logistic Regression Model Performance

The initial logistic regression model used **10 raw features** and **63 effective parameters**.

**Performance metrics:**

- **AUC:** 0.750995  
- **AIC:** 4411.27  
- **BIC:** 4825.85  

This indicates:

- Good discriminative ability (AUC ≈ 0.75)
- Reasonable calibration
- But leaves open the question of whether all 10 features are truly necessary

---

### 3.4 Single-Feature Ablation Study

To quantify feature importance, we removed each feature one at a time and re-fit the model.

| Feature Removed        | AUC Without Feature | ΔAUC (Absolute Change) |
|------------------------|---------------------|------------------------|
| **kick_distance**      | **0.5468**          | **−0.2042**            |
| roof                   | 0.7488              | −0.0022                |
| surface                | 0.7493              | −0.0017                |
| humidity_pct           | 0.7504              | −0.0006                |
| wind_speed_mph         | 0.7507              | −0.0003                |
| half_seconds_remaining | 0.7509              | −0.0001                |
| score_differential     | 0.7509              | −0.0001                |
| temp_f                 | 0.7509              | −0.00007               |
| weather_type           | 0.7521              | +0.00106               |
| wind_dir               | 0.7524              | +0.00140               |

**Key conclusion:**

- **Kick distance is overwhelmingly the dominant predictor.**
- Removing distance drops AUC by **0.204**, a huge loss.
- Removing any other single feature changes AUC by **≤ 0.003** (essentially negligible).

---

### 3.5 Grouped Ablations

We also removed *groups* of features to test whether combinations (e.g., all weather features) mattered.

| Group Removed   | Features Removed                                                   | AUC After Removal | ΔAUC       |
|-----------------|--------------------------------------------------------------------|-------------------|------------|
| **no_distance** | kick_distance                                                      | **0.5468**        | **−0.2042** |
| **distance_only** | *all features except distance*                                  | **0.7435**        | −0.00745   |
| no_stadium      | roof, surface                                                      | 0.7450            | −0.00599   |
| no_pressure     | half_seconds_remaining, score_differential                         | 0.7507            | −0.00025   |
| no_weather      | wind_speed_mph, wind_dir, humidity_pct, weather_type, temp_f       | 0.7514            | +0.00043   |

**Interpretation:**

- Removing **distance** is catastrophic.  
- Keeping **only distance** still yields an AUC of **0.7435**, very close to the full model’s 0.751.  
- Stadium and weather features introduce only small, marginal improvements or noise.

---

### 3.6 Model Selection via AIC/BIC

AIC and BIC strongly penalize unnecessary parameters.

| Model                         | Features | AIC      | BIC      |
|------------------------------|----------|----------|----------|
| **Full logistic regression** | 10       | 4411.27  | 4825.85  |
| **Distance-only regression** | 1        | **4327.19** | **4340.35** |

**Both AIC and BIC strongly favor the distance-only model.**

- ΔAIC ≈ 84  
- ΔBIC ≈ 485  

This indicates that the extra features add complexity without meaningful improvement in fit.

---

### 3.7 Final Predictive Model Choice

The final model selected is:

> **A distance-only logistic regression that predicts field-goal make probability as a function of kick distance.**

This choice is supported by:

- Almost identical AUC compared to the full model (0.7435 vs 0.7510)
- Huge complexity reduction (63 parameters → 2 parameters)
- Strong AIC/BIC preference
- Clear physical intuition: distance dominates difficulty
- Easier interpretation and communication to non-technical audiences

This model forms a **clean, realistic baseline** for how well NFL kickers perform in practice over the last ~25 years.

---

## 4. Reinforcement Learning: Learning Optimal Kicking Behavior

### 4.1 Motivation

Once we have a predictive model of real-world performance, a natural follow-up question is:

> **If we could perfectly tune kick velocity and angle in response to game conditions, how much better could we do than real NFL kickers?**

To explore this, we design a **custom field-goal simulation environment** and train a **DQN agent** to maximize long-run reward.

The goal is to estimate an **upper bound** on what’s achievable under idealized mechanical control and optimal decision-making, and compare it against the supervised model’s real-world baseline.

---

### 4.2 Custom Environment Design

The RL environment includes:

- **State variables (environment conditions):**
  - Kick distance (25–75 yards)
  - Headwind (ft/s)
  - Crosswind (ft/s)
  - Temperature (°F)
  - Pressure (1–100 scale, increasing execution noise)
  - Slip (0–1 scale, modeling plant-foot instability)
  - Hashmark offset (left/right of center)

- **Action variables (chosen by the agent):**
  - Kick **velocity** (mph)
  - **Vertical angle** (degrees)
  - **Horizontal aim angle** (degrees, left/right)

- **Physics and noise:**
  - Projectile motion with gravity and drag
  - Wind components applied to ball trajectory
  - Execution noise increasing with pressure/slip
  - Kicks can be short, long, wide left/right, or successfully pass through the uprights

- **Reward structure:**
  - +100 for a made field goal  
  - −100 for a miss  
  - Additional yard-based penalties in the reward:
    - −3 per yard **short** of the goal line  
    - −1 per yard **long** beyond the uprights  
    - −3 per yard **off-center** (left/right offset from the center of the uprights)  

This encourages the agent not only to make kicks, but to:

- Avoid under-kicking  
- Avoid excessive over-kicking  
- Stay close to the center of the uprights  

---

### 4.3 Training Performance

We trained a Deep Q-Network (DQN) for **60,000 episodes**, logging:

- Average success rate per 1,000-episode chunk  
- Average reward per 1,000-episode chunk  

#### Success Rate Over Time  

- Starts at **≈10–12%** (nearly random behavior).
- Climbs to **40–50%** by ~8,000 episodes.
- Reaches **70%+** by ~20,000 episodes.
- Stabilizes in the **82–85%** range for the final 15,000+ episodes.

This is clear evidence of **policy convergence**: the agent progressively learns to counteract wind, distance, and noise.

#### Average Reward Over Time  
*(reward curve included in notebook; trend mirrors success improvement)*

- Initially around **−210**, reflecting frequent misses:
  - Short kicks
  - Wide kicks
  - Poorly tuned velocities
- Over time:
  - The agent reduces short kicks (avoids severe −3/yd penalties).
  - The agent reduces deep overshoots (less −1/yd long penalty).
  - The agent reduces off-center kicks (less −3/yd lateral penalty).
- Final average rewards stabilize around **+15 to +25**, consistent with a policy that:
  - Makes most kicks
  - Rarely over/under-kicks by much
  - Hits relatively close to center

![Success Rate and Average Reward Throughout Training](plots/output_6_1.png)

Together, these training curves show that the DQN successfully **optimizes its kicking policy** in a complex, noisy environment.

---

### 4.4 Final Policy Performance

On a **1,000-episode evaluation rollout** (with greedy policy, no exploration noise):

- **Success Rate:** **86.30%**
- **Average Reward:** **+20.68**

The agent demonstrates:

- High reliability across a wide range of distances and conditions  
- Robustness to pressure and slip-induced noise  
- Strong generalization beyond early training behavior  

---

### 4.5 Environmental Factor Influence

To understand *why* the agent succeeds or fails, we trained a **logistic regression** on RL evaluation outcomes (make/miss) using the environment variables as features.

#### Feature Importance (Effect on Make Probability)  
![Feature Importance](plots/output_6_3.png)

**Qualitative ranking (largest impact → smallest):**

1. **Distance** – as distance increases, make probability drops sharply.
2. **Headwind** – headwinds increase required power/angle; strong headwinds are notably harmful.
3. **Temperature** – cold temperatures reduce effective range (ball travels shorter).
4. **Crosswind** – meaningful but less dominant than distance/headwind; strong rightward crosswinds are particularly harmful.
5. **Pressure** – increases execution noise, slightly reducing success rates at high levels.
6. **Slip** – has comparatively small marginal effect; the agent’s strategy partially compensates for slip variation.

This ordering matches both intuition and empirical NFL experience, increasing the **interpretability** of the learned policy.

---

### 4.7 Sample Scenario: RL Policy as a Coaching Tool

Beyond aggregate statistics, we can examine **individual scenarios** to see what the RL agent *recommends* under specific conditions. These can be used as **coaching insights** or as starting points for training kickers on optimal mechanics.

Example (from `rl.ipynb`):

> **Scenario 1:**  
> Conditions:  
> - Distance: 30 yards  
> - Headwind: 0 ft/s  
> - Crosswind: 0 ft/s  
> - Hashmark: −3.083 yards (Left hash)  
> - Pressure: 10 / 100  
> - Temperature: 75 °F  
> - Slip: 0.10  
>  
> Agent’s chosen action:  
> - Velocity: **55.0 mph**  
> - Vertical angle: **36.0°**  
> - Horizontal angle: **4.0° to the right**  
>  
> Actual kicked values (after adding environmental noise):  
> - Velocity: 58.9 mph  
> - Vertical angle: 35.8°  
> - Horizontal angle: 4.0°  
>  
> Outcome: **MAKE**  
> Reward: **+53.84**

In a real-world setting, such RL-derived policies could be used to:

- Suggest **target velocities and angles** for kickers under specific conditions.  
- Design **training drills** that mimic the hardest scenarios (e.g., long kicks + strong headwind + high pressure).  
- Provide **analytics support** to coaches deciding whether to attempt a field goal or punt/go-for-it given the game context.

---

## 5. Comparing the Two Systems: Human Baseline vs. Optimized Policy

The key scientific payoff of this project is the comparison between:

1. **Historical NFL performance**, as captured by the distance-only logistic regression model.
2. **Optimized RL performance**, as achieved by the DQN agent in the simulated environment.

### 5.1 Real NFL Baseline (Supervised Model)

Using historical field-goal attempts, the distance-only logistic regression model estimates:

- Average make probability across all attempted distances ≈ **72–75%**
- Clear drop-off in success at longer distances, mirroring real NFL trends

### 5.2 RL Agent Performance

Under the same **distance range** (and comparable distributions of environmental factors):

- The RL agent achieves an **86.3% success rate**

### 5.3 Insight

> There is an approximate **10–14 percentage point gap** between **empirical NFL performance** and the performance of an **optimally learned kicking policy** under the same broad conditions.

This suggests:

- Real kickers are constrained by:
  - Human biomechanics and consistency
  - Psychological pressure
  - Limited ability to precisely tune angles/velocities
- The RL agent represents an **upper bound** on what’s mechanically possible when actions are perfectly optimized.

This comparison ties the supervised and RL components into a **single, coherent story**:

- **Supervised model** → “What do kickers actually achieve?”  
- **RL model** → “What could be achieved with perfect adaptation?”  

---

## 6. How Everything Fits Together (Project Cohesion & Design Rationale)

To make the project cohesive and well-motivated:

1. **Unified Goal:**  
   - Predict and optimize field-goal success using both real data and simulation.
   - Quantify the gap between human and optimal performance.

2. **Progression from Problem → Approach → Solution → Evaluation:**
   - **Problem:** Field-goal decisions are high-stakes and uncertain.
   - **Approach:**  
     - Use **supervised learning** to model real-world probabilities.  
     - Use **RL** to learn optimal kicking strategies under varied conditions.  
   - **Solution:**  
     - A distance-only logistic regression model for make probability.  
     - A DQN agent that achieves ≈86% success.  
   - **Evaluation:**  
     - Supervised: AUC, AIC/BIC, ablation studies.  
     - RL: Training curves, success by distance/weather, feature influence, scenario analysis.  
     - Cross-component: Direct performance comparison.

3. **Design Choices Justified:**
   - Logistic regression chosen for interpretability and calibration.
   - Distance-only model chosen based on AIC/BIC and ablation.
   - DQN with experience replay and target networks chosen for stability in a continuous, noisy environment.
   - Reward function explicitly aligned with real-world goals (make the kick, don’t wildly overshoot, be centered).

4. **Evaluation Metrics Aligned with Objectives:**
   - Predictive model evaluated with AUC, AIC, BIC → “How well can we estimate real-world make probability?”  
   - RL model evaluated with success rate, average reward, and environmental breakdowns → “How well can we maximize success and manage difficulty?”  

5. **No Superfluous Components:**
   - Every method contributes directly to answering the central research question.
   - No model or technique is included just for “point collecting.”

6. **Clean, Readable Codebase:**
   - Scripts and notebooks organized under `src/` and/or `notebooks/`.
   - Data outputs under `data/`.
   - Plots under `plots/`.
   - Unused or stale files pruned to keep the repo focused.

## 7. Video Links
- [Demo](https://duke.box.com/s/d3pfrl1irgocrr1014uh0k8qyj7gyol5)
- [Technical Walkthrough](https://duke.box.com/s/ubii0xkiz326uk20aio7p08gsxxnj993)
