# The Datasets

| **Dataset Name** | **Primary Source** | **Link** | **Data Format** | **FusionCore Versions** |
| --- | --- | --- | --- | --- |
| **NASA C-MAPSS** | NASA PCoE | [Download](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) | Text (`.txt`) | $v_0$ — Research/benchmarking |
| **NASA C-MAPSS** | NASA PCoE | [Download](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) | Text (`.txt`) | $v_1$ — Hybrid model development |
| **NASA N-CMAPSS** | NASA PCoE | [Download](https://data.nasa.gov/dataset/n-cmapss-dataset) | HDF5 (`.h5`) | $v_2$ — Complex dataset |

---

## 1. C-MAPSS Dataset: Operational Settings, Sensors, and Fault Modes

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

Working subsets: **FD001, FD002, FD003, and FD004**.

- **Input:** Multivariate time-series from 21 sensors and 3 operational settings.
- **Target:** Remaining Useful Life (RUL) in cycles.
- **Challenge:** Non-linear degradation under sensor noise and varying operating regimes.

> Source: Saxena, A. and Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Research Centre, Prognostics Data Repository.

---

### 1.2 Operational Settings (3 total)

The three operational settings define the flight envelope condition at each recorded cycle.
FD001 and FD003 operate at a fixed sea-level condition (single regime); FD002 and FD004
span six discrete combinations of altitude, Mach number, and TRA across the full flight
envelope.

| Setting | Physical Parameter | Units | FD001 / FD003 | FD002 / FD004 |
| --- | --- | --- | --- | --- |
| Op1 | Altitude | ft | Constant (0 — sea level) | Variable (0 – 42,000) |
| Op2 | Mach Number | — | Constant (0) | Variable (0 – 0.84) |
| Op3 | Throttle Resolver Angle (TRA) | deg | Constant | Variable (20 – 100) |

---

### 1.3 Sensor Measurements (21 total)

All temperature channels are recorded in degrees Rankine (°R), the absolute thermodynamic
scale referenced to absolute zero on the Fahrenheit interval: T(°R) = T(°F) + 459.67;
T(K) = T(°R) / 1.8. The nominal ISA standard day sea-level ambient temperature of 518.67 °R
corresponds to 288.15 K (15 °C).

| Index | Symbol | Description | Units |
| --- | --- | --- | --- |
| s1 | T2 | Total temperature at fan inlet | °R |
| s2 | T24 | Total temperature at LPC outlet | °R |
| s3 | T30 | Total temperature at HPC outlet | °R |
| s4 | T50 | Total temperature at LPT outlet | °R |
| s5 | P2 | Total pressure at fan inlet | psia |
| s6 | P15 | Total pressure in bypass-duct | psia |
| s7 | P30 | Total pressure at HPC outlet | psia |
| s8 | Nf | Physical fan speed | rpm |
| s9 | Nc | Physical core speed | rpm |
| s10 | epr | Engine pressure ratio (P50/P2) | — |
| s11 | Ps30 | Static pressure at HPC outlet | psia |
| s12 | phi | Ratio of fuel flow to Ps30 | pps/psi |
| s13 | NRf | Corrected fan speed | rpm |
| s14 | NRc | Corrected core speed | rpm |
| s15 | BPR | Bypass ratio | — |
| s16 | farB | Burner fuel-air ratio; governs combustor temperature and combustion efficiency | — |
| s17 | htBleed | Bleed enthalpy; enthalpy of air extracted from the compressor for turbine cooling and pressurisation | — |
| s18 | Nf_dmd | Demanded fan speed; the fan speed commanded by the engine control system (FADEC) | rpm |
| s19 | PCNfR_dmd | Demanded corrected fan speed; the control system's corrected fan speed demand, normalised for inlet pressure and temperature conditions | rpm |
| s20 | W31 | HPT coolant bleed; mass flow rate of cooling air bled to the high-pressure turbine | lbm/s |
| s21 | W32 | LPT coolant bleed; mass flow rate of cooling air bled to the low-pressure turbine | lbm/s |

---

### 1.4 Fault Modes

| Dataset | Fault Mode Classification | Degraded Components | Operating Regimes |
| --- | --- | --- | --- |
| FD001 | Single fault mode | HPC degradation | 1 (sea-level static) |
| FD002 | Single fault mode | HPC degradation | 6 (full flight envelope) |
| FD003 | Multiple fault modes | HPC + Fan degradation | 1 (sea-level static) |
| FD004 | Multiple fault modes | HPC + Fan degradation | 6 (full flight envelope) |

---

### 1.5 Key Dataset Characteristics

| Property | Value |
| --- | --- |
| Training engines — FD001 | 100 |
| Training engines — FD002 | 260 |
| Training engines — FD003 | 100 |
| Training engines — FD004 | 249 |
| **Total training engines** | **709** |
| **Total training rows** | **160,359** |
| Sensors per engine | 21 |
| Operational settings per record | 3 |
| Target variable | RUL (cycles until failure) |


---

### 1.6 Dataset Structure

The four subsets represent a systematic increase in complexity along two independent axes:

**Fault mode complexity:**
- FD001 & FD002 — Single fault mode (HPC degradation only)
- FD003 & FD004 — Multiple simultaneous fault modes (HPC + Fan degradation)

**Operational complexity:**
- FD001 & FD003 — Single operating regime (sea-level static)
- FD002 & FD004 — Six discrete operating regimes across the full flight envelope

This 2×2 structure enables progressive benchmarking: algorithms can be validated on the
simplest single-fault, single-regime condition (FD001) before being evaluated on the most
complex multi-fault, multi-regime condition (FD004).

---

## 2. N-CMAPSS Dataset: Operational Settings, Sensors, and Fault Modes

### 2.1 Operational Settings / Scenario Descriptors (4 total)

| Setting | Description |
| --- | --- |
| Altitude (Alt) | Altitude at takeoff / cruise |
| Flight Mach Number (Mach) | Aircraft speed relative to the speed of sound |
| Throttle Resolver Angle (TRA) | Throttle position setting |
| Total Temperature at Fan Inlet (T2) | Temperature at engine inlet |

---

### 2.2 Measured Signals / Physical Sensors (11 total)

| Sensor # | Symbol | Description | Units |
| --- | --- | --- | --- |
| 1 | P2 | Total pressure at fan inlet | psia |
| 2 | T2 | Total temperature at fan inlet | — |
| 3 | P15 | Pressure at LPC outlet | psia |
| 4 | T24 | Temperature at bypass duct (LPC outlet) | — |
| 5 | P30 | Pressure at HPC outlet | psia |
| 6 | T30 | Temperature at HPC outlet | — |
| 7 | Ps30 | Static pressure at HPC outlet | psia |
| 8 | Wf | Fuel flow rate | — |
| 9 | Nc | High-pressure rotor speed (core) | rpm |
| 10 | Nf | Low-pressure rotor speed (fan) | rpm |
| 11 | EPR | Engine pressure ratio | — |

---

### 2.3 Virtual Sensors (6 total)

| Sensor # | Symbol | Description |
| --- | --- | --- |
| 1 | Nfc | Corrected fan speed |
| 2 | Ncc | Corrected core speed |
| 3 | BPR | Bypass ratio |
| 4 | FAR | Burner fuel-air ratio |
| 5 | HI1 | Engine Health Index 1 |
| 6 | HI2 | Engine Health Index 2 |

---

### 2.4 Fault Modes (7 total)

| Subset | Fault Mode | Degraded Component(s) |
| --- | --- | --- |
| DS01 | Fan degradation | Fan (flow and efficiency loss) |
| DS02 | LPC degradation | Low-pressure compressor |
| DS03 | HPC degradation | High-pressure compressor |
| DS04 | LPT degradation | Low-pressure turbine |
| DS05 | HPT degradation | High-pressure turbine |
| DS06 | Multiple faults | Fan + LPC (compound degradation) |
| DS07 & DS08 | Multiple faults | Multiple components (HPC + LPT and others) |

---

### 2.5 Dataset Structure

| Property | Value |
| --- | --- |
| Total engines | 128 |
| Dataset subsets | 8 (DS01–DS08) |
| Organisation | Development + Test sets per subset |
| File format | HDF5 (`.h5`) |
| Variables per record | 17 (4 operational + 11 measured + 2 RUL/health indicators) |

---

### 2.6 Data Contents per File

Each HDF5 file contains:

1. **Operative Conditions (w)** — 4 scenario-descriptor variables (flight data)
2. **Measured Signals (z_x)** — 11 physical sensor measurements
3. **Virtual Sensors (z_v)** — 6 virtual / derived sensor values
4. **Engine Health Parameters (θ)** — model health parameters and degradation indicators
5. **RUL Label** — Remaining Useful Life in flight cycles
6. **Auxiliary Data** — unit number, flight cycle count, flight class, health state indicators

**Flight classes** are defined by flight length:
- Short flights
- Medium flights
- Long flights (> 10,000 ft altitude)

---

### 2.7 Key Differences from C-MAPSS

| Dimension | C-MAPSS | N-CMAPSS |
| --- | --- | --- |
| Data origin | Purely simulated (T-MATS) | Simulation conditioned on real flight data |
| Fault modes per subset | 1 | Up to 7 |
| Degraded components | 2 (HPC, Fan) | 5 rotating sub-components |
| Virtual sensors | None | 6 |
| Unique engine units | 100–260 per subset | 128 across all subsets |
| Operational variability | Discrete regime combinations | Real commercial flight variability |

N-CMAPSS is significantly more representative of operational prognostics challenges and is
the target dataset for FusionCore $v_2$.

---
