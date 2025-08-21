#!/usr/bin/env python3
"""
eagle_sim.py - Main EAGLE simulation runner
Integrates all components and runs the multi-agent simulation
"""

import time
import numpy as np
import json
import os
import glob
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

from config import PhoenixConfig
from eagle_agent import EAGLEAgent, CollisionAvoidance
from communication import CommunicationNetwork

from world import SimulationWorld


# ------------- new helper -----------------
def planned_uav_count():
    if os.path.exists("waypoints_planning.json"):
        with open("waypoints_planning.json") as f:
            meta = json.load(f).get("meta", {})
            if "total_uavs" in meta:
                return int(meta["total_uavs"])
    # fallback: count per-UAV files
    return len(glob.glob("waypoints_agent__*.json"))


class EAGLESimulation:
    """
    Main simulation class for EAGLE multi-UAV system
    """
    
    def __init__(self,
                 num_agents: Optional[int] = None,
                 mission_duration: float = 3600.0,  # seconds
                 dt: float = 1.0,                   # time step
                 enable_visualization: bool = True):
        
        # ------------- inside EAGLESimulation.__init__ -------------
        if num_agents is None:           # allow None to mean "auto"
            num_agents = planned_uav_count()
        self.num_agents = num_agents
        self.mission_duration = mission_duration
        self.dt = dt
        self.enable_viz = enable_visualization
        self.current_time = 0.0
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.config = PhoenixConfig()
        self.comm_network = CommunicationNetwork()
        self.collision_avoidance = CollisionAvoidance()
        
        # Create agents
        self.agents: List[EAGLEAgent] = []
        self._create_agents()
        
        # Create world
        self.world = SimulationWorld(
            bounds=(0, 10000, 0, 10000),
            use_regions=True,
            enable_weather=True
        )
        
        # Metrics tracking
        self.metrics = {
            "events_detected": 0,
            "events_investigated": 0,
            "thermals_exploited": 0,
            "total_energy_saved": 0.0,
            "collisions": 0,
            "messages_sent": 0
        }
        
        # Visualization
        if self.enable_viz:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
    
    def _setup_logging(self):
        """Configure logging for the simulation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('eagle_sim.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("EAGLE_Sim")
    
    def _create_agents(self):
        """Create and initialize EAGLE agents"""
        # Define starting positions (spread across regions)
        start_positions = [
            [1000, 1000, 0],
            [5000, 5000, 0],
            [9000, 1000, 0],
            [1000, 9000, 0],
            [5000, 1000, 0]
        ]
        
        for i in range(self.num_agents):
            agent_id = f"EAGLE_{i+1}"
            
            # Create agent
            agent = EAGLEAgent(
                cfg=self.config,
                uav_id=agent_id,
                comm_network=self.comm_network,
                home_region=None  # Will assign based on position
            )
            
            # Set initial position
            if i < len(start_positions):
                agent.pos = np.array(start_positions[i], dtype=float)
            else:
                # Random position
                agent.pos = np.array([
                    np.random.uniform(0, 10000),
                    np.random.uniform(0, 10000),
                    0
                ])
            
            self.agents.append(agent)
        
        # Setup waypoints from coverage planner or fallback to defaults
        self._setup_patrol_routes_from_coverage()
        
        # Log agent creation
        for agent in self.agents:
            self.logger.info(f"Created agent {agent.uav_id} at position {agent.pos}")
    
    def _setup_patrol_routes_from_coverage(self):
        """Load waypoints from coverage planner output"""
        import glob
        import json
        
        # Find all agent waypoint files
        waypoint_files = glob.glob("waypoints_agent__*.json")
        
        if not waypoint_files:
            # Fallback to default patterns
            for i, agent in enumerate(self.agents):
                self._setup_patrol_route(agent, i)
            return
        
        # Assign files to agents
        for i, agent in enumerate(self.agents):
            if i < len(waypoint_files):
                # Load waypoints from file
                with open(waypoint_files[i], 'r') as f:
                    data = json.load(f)
                
                # Convert to format expected by agent
                waypoints = [
                    [wp['x'], wp['y'], wp['z']] 
                    for wp in data['waypoints']
                ]
                
                agent.set_waypoints(waypoints)
                
                # Set the home region if available
                if 'mission_info' in data:
                    agent.home_region = data['mission_info'].get('region')
                
                self.logger.info(
                    f"Loaded {len(waypoints)} waypoints for {agent.uav_id} "
                    f"from {waypoint_files[i]}"
                )
            else:
                # More agents than waypoint files - use default
                self._setup_patrol_route(agent, i)
    
    def _setup_patrol_route(self, agent: EAGLEAgent, agent_idx: int):
        """Setup patrol waypoints for an agent"""
        # Create a survey pattern based on agent index
        base_x = 2000 + (agent_idx % 3) * 3000
        base_y = 2000 + (agent_idx // 3) * 3000
        
        waypoints = [
            [base_x, base_y, 400],
            [base_x + 2000, base_y, 400],
            [base_x + 2000, base_y + 2000, 400],
            [base_x, base_y + 2000, 400],
            [base_x, base_y, 400]  # Close the loop
        ]
        
        agent.set_waypoints(waypoints)
        agent.path_loop_mode = "pingpong"  # Enable continuous patrol
    
    def _setup_environment(self):
        """Environment is now handled by world module"""
        self.logger.info(f"World initialized with {len(self.world.regions)} regions")
    
    def run(self):
        """Run the complete EAGLE simulation"""
        self.logger.info(f"Starting EAGLE simulation with {self.num_agents} agents")
        
        # Takeoff all agents
        for agent in self.agents:
            agent.arm()
            agent.takeoff(target_alt_m=400.0)
        
        # Main simulation loop
        step = 0
        start_real_time = time.time()
        
        while self.current_time < self.mission_duration:
            # Keep comm timestamps on simulation time
            self.comm_network.set_sim_time(self.current_time)
            
            # Update environment
            self._update_environment()
            
            # Update each agent
            self._update_agents()
            
            # Process communications
            self.comm_network.process_communications(self.current_time)
            
            # Collision detection
            self._check_collisions()
            
            # Update metrics
            self._update_metrics()
            
            # Visualization
            if self.enable_viz and step % 10 == 0:
                self._update_visualization()
            
            # Log status
            if step % 100 == 0:
                self._log_status()
            
            # Advance time
            self.current_time += self.dt
            step += 1
        
        # Simulation complete
        elapsed_real = time.time() - start_real_time
        self.logger.info(f"Simulation complete in {elapsed_real:.1f}s real time")
        
        # Save results
        self._save_results()
    
    def _update_environment(self):
        """Update world state"""
        self.world.update(self.dt)
    
    def _update_agents(self):
        """Update all agents with collision avoidance"""
        # Get current thermals and events from world
        thermals = self.world.get_active_thermals()
        events = self.world.get_active_events()
        
        # Update wind in config if weather is enabled
        if self.world.weather_enabled:
            self.config.wind_enu = self.world.weather.get_wind_vector()
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Get neighbors for collision avoidance
            neighbors = []
            for j, other in enumerate(self.agents):
                if i != j:
                    neighbors.append({
                        'pos': other.pos,
                        'vel': np.array([
                            other.V_h * np.sin(other.psi),
                            other.V_h * np.cos(other.psi),
                            other.vz
                        ])
                    })
            
            # Calculate collision-free velocity if needed
            if neighbors and agent.flight_mode == "mission":
                current_vel = np.array([
                    agent.V_h * np.sin(agent.psi),
                    agent.V_h * np.cos(agent.psi),
                    agent.vz
                ])
                
                goal_vel = current_vel  # Current desired velocity
                
                safe_vel = self.collision_avoidance.calculate_avoidance_velocity(
                    own_pos=agent.pos,
                    own_vel=current_vel,
                    goal_vel=goal_vel,
                    neighbors=neighbors,
                    obstacles=None,
                    config=self.config,            # use PhoenixConfig limits
                    current_heading=agent.psi,     # radians
                    dt=self.dt
                )
                
                # Apply safe velocity (radians internally)
                if not np.array_equal(safe_vel, goal_vel):
                    safe_speed = float(np.linalg.norm(safe_vel[:2]))
                    if safe_speed > 0.1:
                        agent.psi = float(np.arctan2(safe_vel[0], safe_vel[1]))
                        agent.V_h = safe_speed
                        agent.vz  = float(safe_vel[2])
            
            # Update agent with EAGLE logic
            agent.update(self.dt, self.current_time, thermals, events, self.world)
            
            # Broadcast state periodically
            if self.current_time % 10 < self.dt:  # Every 10 seconds
                agent.broadcast_state()
    
    def _check_collisions(self):
        """Check for any collisions between agents"""
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                dist = np.linalg.norm(agent1.pos - agent2.pos)
                if dist < self.collision_avoidance.d_safe:
                    self.logger.warning(
                        f"NEAR MISS: {agent1.uav_id} and {agent2.uav_id} "
                        f"at distance {dist:.1f}m"
                    )
                    self.metrics["collisions"] += 1
    
    def _update_metrics(self):
        """Update simulation metrics"""
        # Count total events detected/investigated
        all_detected = set()
        all_investigated = set()
        total_thermals = 0
        
        for agent in self.agents:
            all_detected.update(agent.detected_event_ids)
            all_investigated.update(agent.investigated_event_ids)
            total_thermals += len(agent.exploited_thermal_ids)
        
        self.metrics["events_detected"] = len(all_detected)
        self.metrics["events_investigated"] = len(all_investigated)
        self.metrics["thermals_exploited"] = total_thermals
        
        # Communication stats
        comm_stats = self.comm_network.get_network_stats()
        self.metrics["messages_sent"] = comm_stats.get("delivered", 0)
        
        # align with FEnergy: report energy saved from thermals
        self.metrics["total_energy_saved"] = float(sum(a.egain_J for a in self.agents))
        
        # (Optional) also expose motor energy used:
        # self.metrics["total_motor_energy_J"] = float(sum(a.cmotor_J for a in self.agents))
    
    def _update_visualization(self):
        """Update real-time visualization"""
        if not self.enable_viz:
            return
        
        # Use world's visualization
        self.world.visualize(self.ax, show_regions=True)
        
        # Overlay agents
        for agent in self.agents:
            # Agent position
            self.ax.plot(agent.pos[0], agent.pos[1], 'bo', markersize=8)
            
            # Agent ID and status
            status = f"{agent.uav_id}\n{agent.battery_pct():.0f}%\n{agent.soaring_state}"
            self.ax.text(
                agent.pos[0] + 100,
                agent.pos[1],
                status,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7)
            )
            
            # Heading arrow
            arrow_len = 200
            dx = arrow_len * np.sin(agent.psi)
            dy = arrow_len * np.cos(agent.psi)
            self.ax.arrow(
                agent.pos[0], agent.pos[1],
                dx, dy,
                head_width=50,
                head_length=50,
                fc='blue',
                ec='blue'
            )
            
            # Waypoint path
            if agent.waypoints:
                wp_x = [wp[0] for wp in agent.waypoints]
                wp_y = [wp[1] for wp in agent.waypoints]
                self.ax.plot(wp_x, wp_y, 'g--', alpha=0.3)
            
            # Show if investigating
            if agent.event_mode == "investigating" and agent.event_center is not None:
                circle = plt.Circle(
                    agent.event_center,
                    agent.event_radius,
                    fill=False,
                    edgecolor='blue',
                    linewidth=2,
                    linestyle='--'
                )
                self.ax.add_patch(circle)
        
        # Update title with more info
        self.ax.set_title(
            f'EAGLE Simulation - Time: {self.current_time:.0f}s | '
            f'Events: {self.metrics["events_detected"]}D/{self.metrics["events_investigated"]}I | '
            f'Thermals: {self.metrics["thermals_exploited"]}'
        )
        
        plt.pause(0.001)
    
    def _log_status(self):
        """Log current simulation status"""
        self.logger.info(
            f"t={self.current_time:.0f}s | "
            f"Events: {self.metrics['events_detected']}D/{self.metrics['events_investigated']}I | "
            f"Thermals: {self.metrics['thermals_exploited']} | "
            f"Messages: {self.metrics['messages_sent']} | "
            f"Collisions: {self.metrics['collisions']}"
        )
    
    def _save_results(self):
        """Save simulation results"""
        # Save metrics
        with open('eagle_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # (Optional) Write your own CSV export here if you add per-agent history.
        
        # Save communication logs
        self.comm_network.save_message_log()
        
        # Save final snapshot
        snapshot = {
            "simulation_time": self.current_time,
            "metrics": self.metrics,
            "agents": [
                {
                    "id": agent.uav_id,
                    "position": agent.pos.tolist(),
                    "battery_pct": agent.battery_pct(),
                    "events_detected": len(agent.detected_event_ids),
                    "events_investigated": len(agent.investigated_event_ids),
                    "thermals_exploited": len(agent.exploited_thermal_ids)
                }
                for agent in self.agents
            ]
        }
        
        with open('eagle_final_snapshot.json', 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        self.logger.info("Results saved")


# Main execution
if __name__ == "__main__":
    # Create and run simulation
    sim = EAGLESimulation(
        num_agents=None,  # Auto-detect from coverage planner
        mission_duration=1800.0,  # 30 minutes
        dt=1.0,
        enable_visualization=True
    )
    
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Save any partial results
        sim._save_results()
        print("Simulation complete")