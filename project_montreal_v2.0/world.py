#!/usr/bin/env python3
"""
world.py - Complete simulation world for EAGLE testing
Integrates thermals, events, regions, and environmental conditions
"""

import numpy as np
import time
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Import your modules
from thermals import Thermal, spawn_random_thermal, MAX_THERMALS
from events import GroundEvent, RegionEventGenerator, load_region_catalog, spawn_random_event
from config import GEOJSON_PATH, UTM_CRS, PhoenixConfig


@dataclass
class Weather:
    """Weather conditions affecting flight"""
    wind_speed: float = 5.0      # m/s
    wind_direction: float = 270.0  # degrees (from west)
    turbulence: float = 0.1       # 0-1 scale
    visibility: float = 10000.0   # meters
    
    def get_wind_vector(self) -> np.ndarray:
        """Get wind vector in ENU coordinates"""
        # Convert wind FROM direction to wind TO direction
        wind_to = (self.wind_direction + 180) % 360
        wind_rad = np.radians(wind_to)
        return np.array([
            self.wind_speed * np.sin(wind_rad),  # East
            self.wind_speed * np.cos(wind_rad),  # North
            0.0  # Up
        ])


class SimulationWorld:
    """
    Complete simulation environment managing:
    - Thermal field with dynamic generation
    - Event generation using regions
    - Weather conditions
    - Visualization
    """
    
    def __init__(self,
                 bounds: Tuple[float, float, float, float] = (0, 10000, 0, 10000),
                 use_regions: bool = True,
                 enable_weather: bool = True):
        """
        Initialize world
        
        Args:
            bounds: (xmin, xmax, ymin, ymax) in meters
            use_regions: Use Montreal regions for event generation
            enable_weather: Enable weather effects
        """
        self.bounds = bounds
        self.sim_time = 0.0
        self.logger = logging.getLogger("SimWorld")
        
        # Weather
        self.weather_enabled = enable_weather
        self.weather = Weather()
        
        # Regions
        self.use_regions = use_regions
        self.regions = []
        if use_regions:
            try:
                self.regions = load_region_catalog(GEOJSON_PATH, UTM_CRS)
                self.logger.info(f"Loaded {len(self.regions)} regions")
            except Exception as e:
                self.logger.warning(f"Could not load regions: {e}")
                self.use_regions = False
        
        # Thermal field
        self.thermals: List[Thermal] = []
        self.thermal_spawn_rate = 1.0 / 60.0  # Expected thermals per second
        self.max_thermals = MAX_THERMALS
        
        # Event generation
        if self.use_regions and self.regions:
            self.event_generator = RegionEventGenerator(
                regions=self.regions,
                base_events_per_min=6.0,
                alpha=0.5
            )
        else:
            # Fallback to simple generation
            self.event_generator = None
            self.events: List[GroundEvent] = []
        
        # Statistics
        self.stats = {
            "total_thermals_spawned": 0,
            "total_events_spawned": 0,
            "active_thermals": 0,
            "active_events": 0
        }
        
        # Initialize with some content
        self._initialize_world()
    
    def _initialize_world(self):
        """Create initial thermals and events"""
        # Spawn initial thermals
        for _ in range(5):
            self._spawn_thermal()
        
        # Spawn initial events if not using region generator
        if not self.event_generator:
            for _ in range(10):
                event = spawn_random_event(
                    self.bounds,
                    existing_events=self.events
                )
                if event:
                    event.t_gen = self.sim_time  # Use sim time
                    self.events.append(event)
                    self.stats["total_events_spawned"] += 1
    
    def _spawn_thermal(self) -> Optional[Thermal]:
        """Spawn a new thermal"""
        # Use region-aware spawning if available
        if self.use_regions and self.regions:
            thermal = spawn_random_thermal(
                area_bounds=self.bounds,
                existing_thermals=self.thermals,
                now=self.sim_time,
                regions=self.regions,
                weight_mode="density",
                alpha=0.7  # Favor high-density areas
            )
        else:
            thermal = spawn_random_thermal(
                area_bounds=self.bounds,
                existing_thermals=self.thermals,
                now=self.sim_time
            )
        
        if thermal:
            self.thermals.append(thermal)
            self.stats["total_thermals_spawned"] += 1
            self.logger.info(
                f"Spawned thermal at ({thermal.center[0]:.0f}, {thermal.center[1]:.0f}) "
                f"condition={thermal.condition}, radius={thermal.radius:.0f}m"
            )
        
        return thermal
    
    def update(self, dt: float):
        """
        Update world state
        
        Args:
            dt: Time step in seconds
        """
        self.sim_time += dt
        
        # Update thermals
        self._update_thermals(dt)
        
        # Update events
        self._update_events(dt)
        
        # Update weather (if needed)
        if self.weather_enabled:
            self._update_weather(dt)
        
        # Update statistics
        self._update_stats()
    
    def _update_thermals(self, dt: float):
        """Update thermal field"""
        # Remove expired thermals
        self.thermals = [t for t in self.thermals if t.active(self.sim_time)]
        
        # Spawn new thermals probabilistically
        if len(self.thermals) < self.max_thermals:
            if np.random.random() < self.thermal_spawn_rate * dt:
                self._spawn_thermal()
    
    def _update_events(self, dt: float):
        """Update events"""
        if self.event_generator:
            # Use region-based generator
            new_events = self.event_generator.step(
                dt_minutes=dt / 60.0,
                now=self.sim_time
            )
            self.stats["total_events_spawned"] += len(new_events)
        else:
            # Simple generation
            self.events = [e for e in self.events if e.active(self.sim_time)]
            
            # Occasionally spawn new events
            if np.random.random() < 0.01 * dt:  # ~1% chance per second
                event = spawn_random_event(
                    self.bounds,
                    existing_events=self.events
                )
                if event:
                    event.t_gen = self.sim_time
                    self.events.append(event)
                    self.stats["total_events_spawned"] += 1
    
    def _update_weather(self, dt: float):
        """Update weather conditions"""
        # Slowly vary wind
        self.weather.wind_direction += np.random.normal(0, 1.0) * dt
        self.weather.wind_direction = self.weather.wind_direction % 360
        
        self.weather.wind_speed = np.clip(
            self.weather.wind_speed + np.random.normal(0, 0.1) * dt,
            0, 20
        )
    
    def _update_stats(self):
        """Update statistics"""
        self.stats["active_thermals"] = len(self.get_active_thermals())
        
        if self.event_generator:
            self.stats["active_events"] = len(self.event_generator.active_events)
        else:
            self.stats["active_events"] = len([e for e in self.events if e.active(self.sim_time)])
    
    def get_active_thermals(self) -> List[Thermal]:
        """Get currently active thermals"""
        return [t for t in self.thermals if t.active(self.sim_time)]
    
    def get_active_events(self) -> List[GroundEvent]:
        """Get currently active events"""
        if self.event_generator:
            return self.event_generator.active_events
        else:
            return [e for e in self.events if e.active(self.sim_time)]
    
    def get_nearby_thermals(self, pos, max_dist_m):
        """Return active thermals within max_dist_m of pos."""
        nearby = []
        for th in self.get_active_thermals():
            dx = th.center[0] - pos[0]
            dy = th.center[1] - pos[1]
            if (dx*dx + dy*dy)**0.5 <= max_dist_m:
                nearby.append(th)
        return nearby
    
    def get_wind_at_position(self, pos):
        """
        Return horizontal and vertical wind components at pos.
        Now includes thermal updraft if inside a thermal.
        """
        vx, vy, vz = 0.0, 0.0, 0.0
        
        # Base wind
        if self.weather_enabled:
            wind = self.weather.get_wind_vector()
            vx, vy = wind[0], wind[1]
        
        # Add thermal updrafts
        for th in self.get_active_thermals():
            dx = th.center[0] - pos[0]
            dy = th.center[1] - pos[1]
            dist = (dx*dx + dy*dy)**0.5
            if dist <= th.radius and th.base_height <= pos[2] <= th.top_height:
                vz += th.w(pos, self.sim_time)
        
        return vx, vy, vz
    
    def save_snapshot(self, filename: str = "world_snapshot.json"):
        """Save current world state"""
        snapshot = {
            "sim_time": self.sim_time,
            "bounds": self.bounds,
            "weather": {
                "wind_speed": self.weather.wind_speed,
                "wind_direction": self.weather.wind_direction
            },
            "stats": self.stats,
            "thermals": [
                {
                    "center": list(t.center),
                    "radius": t.radius,
                    "strength": t.strength,
                    "condition": t.condition,
                    "remaining_time": t.time_remaining(self.sim_time)
                }
                for t in self.get_active_thermals()
            ],
            "events": [
                {
                    "id": e.id,
                    "level": e.level,
                    "position": [e.cx, e.cy],
                    "radius": e.radius,
                    "age": e.age(self.sim_time)
                }
                for e in self.get_active_events()
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        self.logger.info(f"Saved world snapshot to {filename}")
    
    def visualize(self, ax: plt.Axes, show_regions: bool = True):
        """
        Visualize world state
        
        Args:
            ax: Matplotlib axes
            show_regions: Show region boundaries
        """
        ax.clear()
        
        # Plot regions
        if show_regions and self.regions:
            for region in self.regions:
                # Get exterior coordinates
                if hasattr(region.geom, 'exterior'):
                    x, y = region.geom.exterior.xy
                    ax.plot(x, y, 'k-', alpha=0.2, linewidth=0.5)
                    
                    # Label high-density regions
                    if region.density > 5000:
                        centroid = region.geom.centroid
                        ax.text(centroid.x, centroid.y, region.name[:3],
                               fontsize=8, alpha=0.5, ha='center')
        
        # Plot thermals
        for thermal in self.get_active_thermals():
            color = {'Low': 'lightblue', 'Medium': 'blue', 'High': 'darkblue'}[thermal.condition]
            circle = patches.Circle(
                thermal.center,
                thermal.radius,
                fill=False,
                edgecolor=color,
                linewidth=2,
                alpha=0.6
            )
            ax.add_patch(circle)
            
            # Add strength indicator
            ax.text(thermal.center[0], thermal.center[1], 
                   f"{thermal.strength:.1f}",
                   ha='center', va='center', fontsize=8)
        
        # Plot events
        for event in self.get_active_events():
            color = {'Low': 'yellow', 'Medium': 'orange', 'High': 'red'}[event.level]
            circle = patches.Circle(
                (event.cx, event.cy),
                event.radius,
                fill=True,
                facecolor=color,
                alpha=0.3,
                edgecolor=color,
                linewidth=1
            )
            ax.add_patch(circle)
            
            # Mark high-priority events
            if event.level == 'High':
                ax.plot(event.cx, event.cy, 'r*', markersize=10)
        
        # Wind arrow
        if self.weather_enabled:
            wind = self.weather.get_wind_vector()
            ax.arrow(8500, 9000, wind[0]*100, wind[1]*100,
                    head_width=100, head_length=100,
                    fc='gray', ec='gray', alpha=0.5)
            ax.text(8500, 9500, f"Wind: {self.weather.wind_speed:.1f} m/s",
                   fontsize=10, alpha=0.7)
        
        # Statistics
        stats_text = (
            f"Time: {self.sim_time:.0f}s\n"
            f"Thermals: {self.stats['active_thermals']}/{self.stats['total_thermals_spawned']}\n"
            f"Events: {self.stats['active_events']}/{self.stats['total_events_spawned']}"
        )
        ax.text(100, 9500, stats_text, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Set limits and labels
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('EAGLE Simulation World')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')


class WorldVisualizer:
    """Interactive world visualization"""
    
    def __init__(self, world: SimulationWorld):
        self.world = world
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.paused = False
    
    def animate(self, frame):
        """Animation update function"""
        if not self.paused:
            self.world.update(1.0)  # 1 second updates
        
        self.world.visualize(self.ax)
        return []
    
    def on_key(self, event):
        """Handle key presses"""
        if event.key == ' ':
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
        elif event.key == 's':
            self.world.save_snapshot()
            print("Saved snapshot")
    
    def run(self, interval: int = 1000):
        """Run interactive visualization"""
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  S - Save snapshot")
        print("  Close window to stop")
        
        anim = FuncAnimation(
            self.fig, self.animate,
            interval=interval,  # milliseconds
            blit=False
        )
        
        plt.show()
        return anim


# Testing and demonstration
if __name__ == "__main__":
    print("=== EAGLE World Test ===")
    
    # Create world
    world = SimulationWorld(
        bounds=(0, 10000, 0, 10000),
        use_regions=True,
        enable_weather=True
    )
    
    # Test basic updates
    print("\nRunning 60 second test...")
    for i in range(60):
        world.update(1.0)
        
        if i % 10 == 0:
            print(f"t={i}s: {world.stats}")
    
    # Save snapshot
    world.save_snapshot("test_world.json")
    
    # Run visualization
    print("\nStarting interactive visualization...")
    viz = WorldVisualizer(world)
    viz.run()