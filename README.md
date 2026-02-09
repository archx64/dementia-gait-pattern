## Gait Cycle Parameters Data Dictionary

This document defines the variables used in the gait analysis system. All definitions assume **"Reference Foot"** refers to the side currently being analyzed (e.g., Left Leg).

### 1. Spatial Parameters (Distance)
These parameters measure the physical geometry of the footsteps.

| Variable Name | Unit | Description | Calculation Logic |
| :--- | :--- | :--- | :--- |
| **Stride Length** | cm | The distance traveled between two consecutive heel strikes of the **same foot**. Measures total forward progression including lateral drift. | Euclidean distance between $(x_{start}, z_{start})$ and $(x_{end}, z_{end})$ of the reference heel. |
| **Step Length** | cm | The forward distance along the walking path (Z-axis) between the heel strike of the reference foot and the previous heel strike of the opposite foot. | **Z-axis** difference only: `abs(Reference_Z - Opposite_Z)`. |
| **Step Width** | cm | The lateral (side-to-side) distance between the heels of the two feet during a step. | **X-axis** difference only: `abs(Reference_X - Opposite_X)`. |

### 2. Temporal Parameters (Time)
These parameters measure the timing and rhythm of the walk.

| Variable Name | Unit | Description | Calculation Logic |
| :--- | :--- | :--- | :--- |
| **Walking Speed** | m/s | The average velocity of forward progression. | $(\text{Stride Length [m]}) / (\text{Stride Time [s]})$. |
| **Cadence** | steps/min | The rate of steps taken per minute. | $(60 / \text{Stride Time}) \times 2$. |
| **Stride Time** | s | The duration of one full gait cycle (Heel Strike $\to$ Next Heel Strike of same foot). | $\text{Frame Count} / \text{FPS}$. |
| **Step Time** | s | The time interval between the heel strike of the previous foot and the heel strike of the current foot. | $(\text{Frame}_{Ref\_Strike} - \text{Frame}_{Opp\_Strike}) / \text{FPS}$. |

### 3. Cycle Events (% of Gait Cycle)
These variables mark specific milestones within the 0% (Heel Strike) to 100% (Next Heel Strike) timeline.

| Variable Name | Unit | Description | Normal Value Reference |
| :--- | :--- | :--- | :--- |
| **Opposite Foot Off** | % | The moment the **other foot** leaves the ground. Marks the end of the initial "weight acceptance" phase. | ~10–12% |
| **Opposite Foot Contact** | % | The moment the **other foot** strikes the ground. Marks the end of Single Support. | ~50% |
| **Foot Off (Toe Off)** | % | The moment the **reference foot** leaves the ground. Marks the transition from Stance Phase to Swing Phase. | ~60–62% |

### 4. Cycle Phases (% of Gait Cycle)
These parameters describe duration—how long a specific state lasts relative to the full stride.

| Variable Name | Unit | Description | Normal Value Reference |
| :--- | :--- | :--- | :--- |
| **Single Support** | % | The percentage of time where **only the reference leg** is supporting the body weight. | ~38–40% |
| **Double Support** | % | The total percentage of time where **both feet** are touching the ground simultaneously (Sum of Initial + Terminal Double Support). | ~20–24% |

### 5. Symmetry Indices
These quantify the "limp" or irregularity of the walk.

| Variable Name | Unit | Description | Interpretation |
| :--- | :--- | :--- | :--- |
| **Limp Index** | Ratio | The ratio of Stance Phase time to Swing Phase time. Formula: $\frac{\text{Stance Duration}}{\text{Swing Duration}}$ | **1.0**: Marching (Equal time).<br>**1.5**: Normal walking.<br>**> 1.8**: Limping (Spending too much time on safe leg).<br>**< 1.3**: Antalgic (Painful leg, hopping off it quickly). |