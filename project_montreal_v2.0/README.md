# EAGLE UAV Agent Simulation v2.0

**Enhanced Agent-based Guidance for Low-energy Exploitation**

> **🚀 NEW: Master2 Branch Available!**  
> This branch includes the **Probabilistic Thermal Map Integration** (`probmap` package) that enhances thermal exploitation capabilities using pre-computed probability maps. See [Probabilistic Map Features](#probabilistic-map-features) below.

A sophisticated multi-UAV cooperative decision-making system that implements advanced energy management, thermal soaring, and event investigation capabilities.

## 🚁 Overview

This project implements the EAGLE (Enhanced Agent-based Guidance for Low-energy Exploitation) framework for autonomous UAV operations. The system enables multiple UAVs to cooperatively manage energy resources through thermal soaring while performing surveillance and event investigation tasks.

## ✨ Key Features

### 🎯 **Cooperative Decision Making**
- **Tiered Auction System**: Event investigation auctions with agent health-based priority tiers
- **Thermal Exploitation**: One-shot auctions for thermal soaring opportunities
- **Deterministic Resolution**: Configurable timeouts with consistent tie-breaking

### ⚡ **Advanced Energy Management**
- **Wind-Aware Models**: Aerodynamic calculations considering wind effects and bank angles
- **Thermal Policies**: 
  - `STRICT_ALT`: Motor-off savings only
  - `ALT_BAND`: Altitude gain within defined bands
- **Robust Thermal Detection**: Confidence-based thermal identification across agents

### 🛡️ **Safety & Navigation**
- **Collision Avoidance**: Artificial Potential Field (APF) with velocity projection
- **Deterministic Communication**: Simulation-time-based message handling
- **Modular Architecture**: Standalone agents with complete flight capabilities

### 🗺️ **Probabilistic Map Features**
- **Offline Map Generation**: Pre-computed conditional probability maps for thermal events
- **Context-Aware Queries**: Season, time-of-day, and land cover based thermal probability estimation
- **Fallback Strategy**: Automatic fallback to probabilistic map when live thermal opportunities are scarce
- **Geospatial Filtering**: Region-aware thermal search within patrol sectors and adjacent areas

## 📁 Project Structure

```
project_montreal_v2.0/
├── eagle_agent.py          # Core EAGLE agent implementation
├── eagle_sim.py           # Main simulation orchestrator
├── communication.py       # Inter-agent communication network
├── config.py             # Configuration parameters
├── world.py              # Simulation world and environment
├── thermals.py           # Thermal modeling (external dependency)
├── events.py             # Ground event modeling (external dependency)
├── probmap/              # Probabilistic thermal map package
│   ├── __init__.py       # Package initialization
│   ├── offline.py        # Offline map generation and building
│   └── online.py         # Online probability map querying
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NumPy
- Matplotlib
- Required dependencies: `thermals.py`, `events.py`

### Installation
```bash
git clone https://github.com/mirzaanaswahid/project_montreal_v2.0.git
cd project_montreal_v2.0
```

### Running the Simulation
```bash
python eagle_sim.py
```

## 🔧 Configuration

Key parameters can be adjusted in `config.py`:

```python
@dataclass
class PhoenixConfig:
    # EAGLE-specific parameters
    eagle_tier_timeout_s: float = 5.0
    eagle_auction_timeout_s: float = 20.0
    eagle_benefit_threshold_j: float = 500.0
    
    # Aerodynamic parameters
    air_density: float = 1.225
    CD0: float = 0.012
    induced_drag_k: float = 0.04
    prop_eta: float = 0.85
    
    # Operational parameters
    altitude_ref: float = 400.0
    alt_band_m: float = 100.0
    V_range_opt: float = 18.0
    V_loiter: float = 15.0
    
    # Probability map configuration
    probmap_meta_path: str = "path/to/probability_map_metadata.json"
    probmap_prob_path: str = "path/to/conditional_probability_map.npy"
    probmap_avg_npz_path: str = "path/to/average_thermal_metrics.npz"
    probmap_lc_raster_path: str = "path/to/landcover_classification.tif"
    
    # Runtime knobs
    probmap_aoi_half_width_m: float = 500.0
    probmap_probability_threshold: float = 0.5

## 🎮 Simulation Features

### **Agent Capabilities**
- **Flight Modes**: Ground, Armed, Takeoff, Mission, Landing
- **Soaring States**: Normal, Thermal Exploitation, Gliding
- **Event Modes**: Idle, Investigating

### **Communication**
- **Message Types**: Handover requests, bids, task claims, thermal discoveries
- **Network**: Deterministic simulation-time-based messaging
- **Broadcasting**: Periodic state updates and discoveries

### **Visualization**
- **Real-time Display**: Agent positions, headings, battery levels
- **Event Tracking**: Investigation circles and thermal locations
- **Metrics**: Events detected/investigated, thermals exploited, collisions

## 🔬 Technical Implementation

### **EAGLE Algorithm**
1. **Safety Check**: Critical battery levels trigger emergency landing
2. **High-Priority Events**: Immediate investigation of urgent events
3. **Handover Requests**: Cooperative task assignment through auctions
4. **Energy Management**: Proactive thermal seeking when battery low
5. **Discovery Announcements**: Sharing thermal and event information

### **Auction Mechanisms**
- **Event Auctions**: Tiered bidding based on agent health
- **Thermal Auctions**: One-shot benefit-based competition
- **Deterministic Resolution**: Timeout-based with agent ID tie-breaking

### **Physics Integration**
- **Position Updates**: Wind-corrected velocity integration
- **Energy Consumption**: Power modeling with thermal savings
- **Flight Control**: Heading and speed command execution

## 📊 Output & Results

The simulation generates:
- **Metrics JSON**: `eagle_metrics.json` with performance statistics
- **Communication Logs**: Message history and network statistics
- **Final Snapshot**: `eagle_final_snapshot.json` with agent states
- **Real-time Visualization**: Matplotlib-based simulation display

## 🤝 Contributing

This project is part of the EAGLE UAV research framework. For contributions or questions, please contact the development team.

## 📄 License

This project is developed for research purposes in autonomous UAV systems.

---

**Developed by**: EAGLE Research Team  
**Repository**: https://github.com/mirzaanaswahid/project_montreal_v2.0  
**Branches**: 
- `master`: Original EAGLE implementation
- `master2`: **NEW** - Includes probabilistic thermal map integration  
**Version**: 2.0 (with probmap enhancement) 