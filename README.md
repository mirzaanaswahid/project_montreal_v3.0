# Project Montreal v3.0 - EAGLE UAV System

## Overview
This repository contains the EAGLE (Efficient Autonomous Glider for Long-range Exploration) UAV system implementation for Montreal thermal detection and event coordination. The system features distributed multi-agent coordination, thermal soaring capabilities, and intelligent event response.

## Features
- **Multi-Agent Coordination**: Distributed UAV fleet management with handover protocols
- **Thermal Detection**: Advanced thermal soaring algorithms for energy-efficient flight
- **Event Response**: Intelligent event detection and response coordination
- **Patrol Management**: Automated patrol sector management and handover
- **Communication Network**: Perfect communication system with priority-based messaging
- **Montreal Integration**: Customized for Montreal's geographic and environmental conditions

## Project Structure
```
project_montreal_v2.0/
├── communication.py          # Inter-agent messaging system
├── eagle_agent.py           # Main UAV agent implementation
├── eagle_sim.py             # Simulation environment
├── thermals.py              # Thermal detection and soaring logic
├── event_detection.py       # Event detection algorithms
├── patrol_sector.py         # Patrol area management
├── config.py                # Configuration parameters
├── world.py                 # World/environment representation
├── probmap/                 # Probability mapping modules
│   ├── offline.py          # Offline probability calculations
│   └── online.py           # Real-time probability updates
└── files/                   # Data files and resources
```

## Key Components

### Communication System
- Perfect communication network with no packet loss
- Priority-based message ordering
- Support for all EAGLE message types (handover requests, thermal bids, etc.)
- Range-based connectivity

### EAGLE Agent
- Autonomous decision making
- Thermal soaring capabilities
- Event response coordination
- Patrol management
- Health monitoring

### Thermal Detection
- Synthetic thermal data generation
- Real-time thermal mapping
- Energy-efficient flight planning
- Thermal strength and radius analysis

### Event Coordination
- Distributed event bidding
- Task claiming and handover
- Fallback investigation protocols
- Collision avoidance alerts

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- Matplotlib
- GeoPandas (for geographic data)
- ROS2 (optional, for hardware integration)

### Setup
```bash
# Clone the repository
git clone https://github.com/mirzaanaswahid/project_montreal_v3.0.git
cd project_montreal_v3.0

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Simulation
```bash
cd project_montreal_v2.0
python eagle_sim.py
```

### Testing Communication
```bash
python communication.py
```

### Individual Components
```bash
# Test thermal detection
python thermals.py

# Test event detection
python event_detection.py

# Test patrol management
python patrol_sector.py
```

## Configuration
Edit `config.py` to modify:
- UAV fleet size and parameters
- Communication ranges
- Thermal detection thresholds
- Patrol area definitions
- Simulation parameters

## Data Files
**Note**: Large data files (>100MB) are excluded from this repository due to GitHub limits:
- `conditional_probability_map.npy` (112.95 MB)
- `landcover-2020-classification.tif` (2010.07 MB)

These files can be obtained separately or regenerated using the provided scripts.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
- **Author**: Mirza Anas Wahid
- **GitHub**: [@mirzaanaswahid](https://github.com/mirzaanaswahid)

## Acknowledgments
- EAGLE research team
- Montreal research community
- UAV thermal soaring research
- Distributed systems research

## Version History
- **v3.0**: Current version with enhanced communication and coordination
- **v2.0**: Previous version with basic thermal detection
- **v1.0**: Initial prototype

---
*Built with ❤️ for autonomous UAV research and Montreal's future* 