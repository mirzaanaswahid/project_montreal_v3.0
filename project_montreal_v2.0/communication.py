#!/usr/bin/env python3
"""
communication.py - Inter-agent messaging system for EAGLE
Handles all agent-to-agent communications with perfect delivery (no loss)
"""
import time
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import numpy as np

class MessageType(Enum):
    """EAGLE message types from the formulation"""
    # Event-related
    HANDOVER_REQUEST = "HANDOVER_REQUEST"
    EVENT_BID = "EVENT_BID"
    TASK_CLAIMED = "TASK_CLAIMED"
    FALL_BACK_INVESTIGATING = "FALL_BACK_INVESTIGATING"
    
    # Thermal-related
    THERMAL_DISCOVERED = "THERMAL_DISCOVERED"
    LIVE_THERMAL_BID = "LIVE_THERMAL_BID"
    THERMAL_CLAIMED = "THERMAL_CLAIMED"
    
    # Patrol Coordination
    PATROL_HANDOVER_REQUEST = "PATROL_HANDOVER_REQUEST"
    PATROL_RESUME_NOTICE = "PATROL_RESUME_NOTICE"
    
    # Safety
    COLLISION_ALERT = "COLLISION_ALERT"
    
    # State sharing
    AGENT_STATE = "AGENT_STATE"
    HEARTBEAT = "HEARTBEAT"

@dataclass
class Message:
    """Standard message format for EAGLE communications"""
    msg_id: str
    msg_type: MessageType
    sender_id: str
    timestamp: float
    data: Dict[str, Any]
    ttl: float = 30.0  # Time to live in seconds
    recipients: Optional[Set[str]] = None  # None = broadcast
    priority: int = 0  # 0=normal, 1=high, 2=critical
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "ttl": self.ttl,
            "recipients": list(self.recipients) if self.recipients else None,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Message':
        """Create from dictionary"""
        return cls(
            msg_id=d["msg_id"],
            msg_type=MessageType(d["msg_type"]),
            sender_id=d["sender_id"],
            timestamp=d["timestamp"],
            data=d["data"],
            ttl=d.get("ttl", 30.0),
            recipients=set(d["recipients"]) if d.get("recipients") else None,
            priority=d.get("priority", 0)
        )

class MessageFactory:
    """Factory class for creating typed messages"""
    
    @staticmethod
    def create_handover_request(sender_id: str, event_id: str, event_position: List[float], 
                                event_type: str, priority: float, reason: str, timestamp: float) -> Message:
        """Create HANDOVER_REQUEST message"""
        return Message(
            msg_id="",
            msg_type=MessageType.HANDOVER_REQUEST,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "event_id": event_id,
                "event_position": event_position,
                "event_type": event_type,
                "priority": priority,
                "reason": reason,
                "timestamp": timestamp
            },
            priority=1  # High priority
        )
    
    @staticmethod
    def create_event_bid(sender_id: str, event_id: str, bid_cost: float, tier: str,
                         current_position: List[float], timestamp: float) -> Message:
        """Create EVENT_BID message"""
        return Message(
            msg_id="",
            msg_type=MessageType.EVENT_BID,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "event_id": event_id,
                "bid_cost": bid_cost,
                "tier": tier,
                "current_position": current_position,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_task_claimed(sender_id: str, task_id: str, task_type: str,
                            winning_bid: float, timestamp: float) -> Message:
        """Create TASK_CLAIMED message"""
        return Message(
            msg_id="",
            msg_type=MessageType.TASK_CLAIMED,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "task_id": task_id,
                "task_type": task_type,
                "claimed_by": sender_id,
                "winning_bid": winning_bid,
                "timestamp": timestamp
            },
            priority=1
        )
    
    @staticmethod
    def create_fall_back_investigating(sender_id: str, event_id: str, position: List[float],
                                       reason: str, timestamp: float) -> Message:
        """Create FALL_BACK_INVESTIGATING message"""
        return Message(
            msg_id="",
            msg_type=MessageType.FALL_BACK_INVESTIGATING,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "event_id": event_id,
                "position": position,
                "reason": reason,
                "timestamp": timestamp
            },
            priority=2  # Critical
        )
    
    @staticmethod
    def create_thermal_discovered(sender_id: str, thermal_id: str, position: List[float],
                                  strength: float, radius: float, timestamp: float) -> Message:
        """Create THERMAL_DISCOVERED message"""
        return Message(
            msg_id="",
            msg_type=MessageType.THERMAL_DISCOVERED,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "thermal_id": thermal_id,
                "position": position,
                "strength": strength,
                "radius": radius,
                "discoverer_id": sender_id,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_live_thermal_bid(sender_id: str, thermal_id: str, bid_score: float,
                                current_position: List[float], battery_level: float, 
                                timestamp: float) -> Message:
        """Create LIVE_THERMAL_BID message"""
        return Message(
            msg_id="",
            msg_type=MessageType.LIVE_THERMAL_BID,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "thermal_id": thermal_id,
                "bid_score": bid_score,
                "current_position": current_position,
                "battery_level": battery_level,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_thermal_claimed(sender_id: str, thermal_id: str, investigation_start: float,
                               timestamp: float) -> Message:
        """Create THERMAL_CLAIMED message"""
        return Message(
            msg_id="",
            msg_type=MessageType.THERMAL_CLAIMED,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "thermal_id": thermal_id,
                "claimed_by": sender_id,
                "investigation_start": investigation_start,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_patrol_handover_request(sender_id: str, patrol_area_id: str, 
                                       current_waypoints: List[List[float]], reason: str,
                                       task_position: List[float], timestamp: float) -> Message:
        """Create PATROL_HANDOVER_REQUEST message"""
        return Message(
            msg_id="",
            msg_type=MessageType.PATROL_HANDOVER_REQUEST,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "requesting_agent_id": sender_id,
                "patrol_area_id": patrol_area_id,
                "current_waypoints": current_waypoints,
                "reason": reason,
                "task_position": task_position,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_patrol_resume_notice(sender_id: str, patrol_area_id: str,
                                    current_position: List[float], timestamp: float) -> Message:
        """Create PATROL_RESUME_NOTICE message"""
        return Message(
            msg_id="",
            msg_type=MessageType.PATROL_RESUME_NOTICE,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "agent_id": sender_id,
                "patrol_area_id": patrol_area_id,
                "current_position": current_position,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_collision_alert(sender_id: str, target_agent_id: str, alerting_position: List[float],
                               distance: float, failure_type: str, timestamp: float) -> Message:
        """Create COLLISION_ALERT message"""
        return Message(
            msg_id="",
            msg_type=MessageType.COLLISION_ALERT,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "alerting_agent_id": sender_id,
                "target_agent_id": target_agent_id,
                "alerting_position": alerting_position,
                "distance": distance,
                "severity": "critical",
                "failure_type": failure_type,
                "timestamp": timestamp
            },
            recipients={target_agent_id} if target_agent_id != "broadcast" else None,
            priority=2  # Critical priority
        )
    
    @staticmethod
    def create_agent_state(sender_id: str, position: List[float], velocity: List[float],
                          battery_level: float, current_task: str, task_id: str,
                          state: str, health_status: str, tier_status: str, 
                          timestamp: float) -> Message:
        """Create AGENT_STATE message"""
        return Message(
            msg_id="",
            msg_type=MessageType.AGENT_STATE,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "agent_id": sender_id,
                "position": position,
                "velocity": velocity,
                "battery_level": battery_level,
                "current_task": current_task,
                "task_id": task_id,
                "state": state,
                "health_status": health_status,
                "tier_status": tier_status,
                "timestamp": timestamp
            }
        )
    
    @staticmethod
    def create_heartbeat(sender_id: str, position: List[float], battery_level: float,
                        operational: bool, timestamp: float) -> Message:
        """Create HEARTBEAT message"""
        return Message(
            msg_id="",
            msg_type=MessageType.HEARTBEAT,
            sender_id=sender_id,
            timestamp=timestamp,
            data={
                "agent_id": sender_id,
                "position": position,
                "battery_level": battery_level,
                "operational": operational,
                "timestamp": timestamp
            }
        )

class CommunicationNetwork:
    """
    Perfect UAV communication network with:
    - No packet loss (100% delivery)
    - Range-based connectivity
    - Minimal delays
    - Priority-based message ordering
    - Message history for debugging
    """
    
    def __init__(self, 
                 comm_range: float = 5000.0,  # meters
                 base_delay: float = 0.01,     # 10ms base delay for perfect comm
                 enable_logging: bool = True):
        
        self.comm_range = comm_range
        self.base_delay = base_delay
        
        # Message queues for each agent (priority queues)
        self.inbox: Dict[str, deque] = defaultdict(deque)
        self.outbox: Dict[str, List[Message]] = defaultdict(list)
        
        # Agent positions for range checking
        self.agent_positions: Dict[str, np.ndarray] = {}
        
        # Message tracking
        self.message_history: List[Message] = []
        self.message_counter = 0
        
        # Message factory
        self.factory = MessageFactory()
        
        # Statistics tracking
        self.stats = {
            "posted": 0,
            "delivered": 0,
            "out_of_range": 0,
            "expired": 0
        }
        
        # Simulation clock
        self.sim_time: float = 0.0
        
        # Logging
        if enable_logging:
            self.logger = logging.getLogger("EAGLE_Comms")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                self.logger.addHandler(handler)
        else:
            self.logger = None
    
    def register_agent(self, agent_id: str, position: np.ndarray):
        """Register an agent with the network"""
        self.agent_positions[agent_id] = position.copy()
        if agent_id not in self.inbox:
            self.inbox[agent_id] = deque()
        self._log(f"Agent {agent_id} registered at position {position}")
    
    def update_agent_position(self, agent_id: str, position: np.ndarray):
        """Update agent position for range calculations"""
        if agent_id in self.agent_positions:
            self.agent_positions[agent_id] = position.copy()
    
    def set_sim_time(self, t: float) -> None:
        """Set the current simulation time (seconds)."""
        self.sim_time = float(t)
    
    def send_message(self, msg: Message):
        """Queue a message for delivery with perfect communication"""
        self.message_counter += 1
        msg.msg_id = f"{msg.sender_id}_{self.message_counter:06d}"
        msg.timestamp = self.sim_time  # Ensure timestamp is current sim time
        
        # Store in history
        self.message_history.append(msg)
        
        # Add to outbox for processing
        self.outbox[msg.sender_id].append(msg)
        
        # Increment posted counter
        self.stats["posted"] += 1
        
        self._log(f"Message queued: {msg.msg_type.value} from {msg.sender_id} (priority={msg.priority})")
    
    def broadcast(self, sender_id: str, msg_type: MessageType, data: Dict[str, Any], priority: int = 0):
        """Convenience method for broadcasting"""
        msg = Message(
            msg_id="",
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=self.sim_time,
            data=data,
            recipients=None,
            priority=priority
        )
        self.send_message(msg)
    
    def send_to(self, sender_id: str, recipient_id: str, msg_type: MessageType, 
                data: Dict[str, Any], priority: int = 0):
        """Send message to specific recipient"""
        msg = Message(
            msg_id="",
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=self.sim_time,
            data=data,
            recipients={recipient_id},
            priority=priority
        )
        self.send_message(msg)
    
    def process_communications(self, current_time: float):
        """
        Process all pending communications with perfect delivery
        Called once per simulation step
        """
        # Update simulation time
        self.sim_time = current_time
        
        # Process each agent's outbox
        for sender_id, messages in self.outbox.items():
            if sender_id not in self.agent_positions:
                continue
                
            sender_pos = self.agent_positions[sender_id]
            
            # Sort messages by priority (higher priority first)
            messages.sort(key=lambda m: -m.priority)
            
            for msg in messages:
                # Check TTL
                if current_time - msg.timestamp > msg.ttl:
                    self._log(f"Message expired: {msg.msg_id}")
                    self.stats["expired"] += 1
                    continue
                
                # Determine recipients
                if msg.recipients:
                    recipients = msg.recipients
                else:
                    # Broadcast - all agents except sender
                    recipients = set(self.agent_positions.keys()) - {sender_id}
                
                # Deliver to each recipient with perfect communication
                for recipient_id in recipients:
                    if recipient_id not in self.agent_positions:
                        continue
                    
                    # Check range
                    recipient_pos = self.agent_positions[recipient_id]
                    distance = np.linalg.norm(sender_pos - recipient_pos)
                    
                    if distance > self.comm_range:
                        self._log(f"Out of range: {sender_id} -> {recipient_id} ({distance:.0f}m)")
                        self.stats["out_of_range"] += 1
                        continue
                    
                    # Calculate minimal delay based on distance (perfect communication)
                    delay = self.base_delay + (distance / self.comm_range) * 0.01  # Max 20ms delay
                    delivery_time = msg.timestamp + delay
                    
                    # Add to recipient's inbox with delivery time (guaranteed delivery)
                    self.inbox[recipient_id].append((delivery_time, msg))
                    
                    self._log(f"Message scheduled for delivery: {msg.msg_type.value} to {recipient_id} at t={delivery_time:.3f}")
        
        # Clear outboxes
        self.outbox.clear()
    
    def get_messages(self, agent_id: str, current_time: float) -> List[Message]:
        """Get all messages that have arrived for an agent"""
        if agent_id not in self.inbox:
            return []
        
        delivered = []
        remaining = deque()
        
        # Check each message in inbox
        while self.inbox[agent_id]:
            delivery_time, msg = self.inbox[agent_id].popleft()
            
            if delivery_time <= current_time:
                delivered.append(msg)
                self.stats["delivered"] += 1
                self._log(f"Delivered: {msg.msg_type.value} to {agent_id}")
            else:
                # Not ready yet, keep in queue
                remaining.append((delivery_time, msg))
        
        # Put back messages not yet delivered
        self.inbox[agent_id] = remaining
        
        # Sort delivered messages by priority then timestamp
        delivered.sort(key=lambda m: (-m.priority, m.timestamp))
        
        return delivered
    
    def get_messages_by_type(self, agent_id: str, msg_type: MessageType, 
                            current_time: float) -> List[Message]:
        """Get messages of specific type for an agent"""
        all_messages = self.get_messages(agent_id, current_time)
        return [msg for msg in all_messages if msg.msg_type == msg_type]
    
    def check_connectivity(self, agent1_id: str, agent2_id: str) -> bool:
        """Check if two agents are within communication range"""
        if agent1_id not in self.agent_positions or agent2_id not in self.agent_positions:
            return False
        
        pos1 = self.agent_positions[agent1_id]
        pos2 = self.agent_positions[agent2_id]
        distance = np.linalg.norm(pos1 - pos2)
        
        return distance <= self.comm_range
    
    def get_neighbors(self, agent_id: str, max_range: Optional[float] = None) -> List[str]:
        """Get all agents within communication range"""
        if agent_id not in self.agent_positions:
            return []
        
        if max_range is None:
            max_range = self.comm_range
            
        agent_pos = self.agent_positions[agent_id]
        neighbors = []
        
        for other_id, other_pos in self.agent_positions.items():
            if other_id == agent_id:
                continue
            distance = np.linalg.norm(agent_pos - other_pos)
            if distance <= max_range:
                neighbors.append(other_id)
        
        return neighbors
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        total_sent = len(self.message_history)
        total_pending = sum(len(inbox) for inbox in self.inbox.values())
        
        msg_type_counts = defaultdict(int)
        for msg in self.message_history:
            msg_type_counts[msg.msg_type.value] += 1
        
        # Calculate delivery rate (perfect communication should be near 100%)
        delivery_rate = 0.0
        if self.stats["posted"] > 0:
            successful = self.stats["delivered"]
            attempted = self.stats["posted"] - self.stats["expired"]
            if attempted > 0:
                delivery_rate = (successful / attempted) * 100
        
        stats = dict(self.stats)
        stats.update({
            "total_messages_sent": total_sent,
            "messages_pending_delivery": total_pending,
            "message_types": dict(msg_type_counts),
            "agents_connected": len(self.agent_positions),
            "delivery_rate": f"{delivery_rate:.1f}%"
        })
        
        return stats
    
    def save_message_log(self, filename: str = "eagle_comms.json"):
        """Save message history to file"""
        log_data = {
            "stats": self.get_network_stats(),
            "messages": [msg.to_dict() for msg in self.message_history[-1000:]]  # Last 1000
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self._log(f"Saved message log to {filename}")
    
    def clear_expired_messages(self, current_time: float):
        """Remove expired messages from all inboxes"""
        for agent_id in self.inbox:
            remaining = deque()
            while self.inbox[agent_id]:
                delivery_time, msg = self.inbox[agent_id].popleft()
                if current_time - msg.timestamp <= msg.ttl:
                    remaining.append((delivery_time, msg))
                else:
                    self.stats["expired"] += 1
            self.inbox[agent_id] = remaining
    
    def _log(self, message: str):
        """Internal logging"""
        if self.logger:
            self.logger.info(message)

# Example usage and testing
if __name__ == "__main__":
    # Create network with perfect communication
    network = CommunicationNetwork(
        comm_range=5000.0,
        base_delay=0.01,  # 10ms base delay
        enable_logging=True
    )
    
    # Register agents
    network.register_agent("UAV1", np.array([0, 0, 400]))
    network.register_agent("UAV2", np.array([1000, 1000, 400]))
    network.register_agent("UAV3", np.array([2000, 2000, 400]))
    network.register_agent("UAV4", np.array([3000, 3000, 400]))
    
    # Set initial time
    t0 = 0.0
    network.set_sim_time(t0)
    
    # Test various message types using the factory
    
    # 1. Event coordination
    msg1 = network.factory.create_handover_request(
        "UAV1", "evt_001", [500, 500], "anomaly", 0.9, "low_battery", t0
    )
    network.send_message(msg1)
    
    # 2. Thermal discovery
    msg2 = network.factory.create_thermal_discovered(
        "UAV2", "th_100_200", [100, 200], 3.5, 100, t0
    )
    network.send_message(msg2)
    
    # 3. Agent state update
    msg3 = network.factory.create_agent_state(
        "UAV3", [2000, 2000, 400], [10, 5, 0], 75.0,
        "patrolling", "", "NOMINAL", "HEALTHY", "HEALTHY_NOMAD", t0
    )
    network.send_message(msg3)
    
    # 4. Critical collision alert
    msg4 = network.factory.create_collision_alert(
        "UAV1", "UAV2", [500, 500, 400], 150.0, "sensor_failure", t0
    )
    network.send_message(msg4)
    
    # Process communications
    network.process_communications(t0)
    
    # Advance time and retrieve messages
    t1 = 0.02  # 20ms later
    network.set_sim_time(t1)
    
    print("\n" + "="*60)
    print("EAGLE Communication Network Test - Perfect Communication")
    print("="*60)
    
    for agent_id in ["UAV1", "UAV2", "UAV3", "UAV4"]:
        messages = network.get_messages(agent_id, t1)
        if messages:
            print(f"\n{agent_id} received {len(messages)} messages:")
            for msg in messages:
                print(f"  - {msg.msg_type.value} from {msg.sender_id} (priority={msg.priority})")
    
    print("\n" + "-"*60)
    print("Network Statistics:")
    print("-"*60)
    stats = network.get_network_stats()
    for key, value in stats.items():
        if key != "message_types":
            print(f"  {key}: {value}")
    
    print("\n  Message type breakdown:")
    for msg_type, count in stats["message_types"].items():
        print(f"    {msg_type}: {count}")
    
    # Test neighbor detection
    print("\n" + "-"*60)
    print("Connectivity Test:")
    print("-"*60)
    for agent_id in ["UAV1", "UAV2", "UAV3", "UAV4"]:
        neighbors = network.get_neighbors(agent_id)
        print(f"  {agent_id} neighbors: {neighbors}")
    
    # Save log
    network.save_message_log("test_eagle_comms.json")
    print("\n✓ Message log saved to test_eagle_comms.json")