"""
MAGMA - Motion-Activated Generative MIDI Application
Professional blob detection to MIDI conversion system
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import cv2
import PIL.Image, PIL.ImageTk
import mido
import numpy as np
import math
import time
import random
import json
import os
from collections import defaultdict
from threading import Lock
from typing import Optional, Dict, List, Tuple

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Minor": [0, 2, 3, 5, 7, 8, 10],
    "Minor Pentatonic": [0, 3, 5, 7, 10],
    "Major Pentatonic": [0, 2, 4, 7, 9],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
}

CHORD_MASKS = {
    "Full Scale": [0, 1, 2, 3, 4, 5, 6],
    "Triad (1-3-5)": [0, 2, 4],
    "Seventh (1-3-5-7)": [0, 2, 4, 6],
    "Open 5ths (1-5)": [0, 4],
}

SIZE_PRESETS = {
    "All": (0, 10000),
    "Tiny": (50, 500),
    "Small": (500, 1500),
    "Medium": (1500, 3000),
    "Large": (3000, 10000)
}

ACCENT_CHANNEL = 15

# State-based blob colors (BGR format for OpenCV)
BLOB_COLOR_OFF = (0, 255, 0)        # Green - detected, idle
BLOB_COLOR_ATTACK = (0, 165, 255)   # Orange - building up
BLOB_COLOR_ON = (0, 0, 255)         # Red - actively playing
BLOB_COLOR_RELEASE = (0, 255, 255)  # Yellow - releasing

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def interpolate_color(color1: tuple, color2: tuple, t: float) -> tuple:
    """Linear interpolation between two RGB/BGR colors.

    Args:
        color1: (B, G, R) or (R, G, B) start color (0-255)
        color2: (B, G, R) or (R, G, B) end color (0-255)
        t: Interpolation factor (0.0-1.0)

    Returns:
        tuple: Interpolated color (same format as input)
    """
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
    c1 = int(color1[0] + (color2[0] - color1[0]) * t)
    c2 = int(color1[1] + (color2[1] - color1[1]) * t)
    c3 = int(color1[2] + (color2[2] - color1[2]) * t)
    return (c1, c2, c3)

def note_to_name(note_num: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')"""
    note_name = NOTES[note_num % 12]
    octave = (note_num // 12) - 1
    return f"{note_name}{octave}"

def name_to_note(note_name: str) -> int:
    """Convert note name to MIDI number (e.g., 'C4' -> 60)"""
    note_name = note_name.strip()
    if not note_name:
        return 60
    
    note_part = ""
    octave_part = ""
    
    if len(note_name) >= 2:
        if note_name[1] in ['#', 'b']:
            note_part = note_name[:2]
            octave_part = note_name[2:]
        else:
            note_part = note_name[0]
            octave_part = note_name[1:]
    else:
        note_part = note_name[0]
        octave_part = "4"
    
    try:
        if note_part in NOTES:
            note_idx = NOTES.index(note_part)
        elif note_part in NOTE_NAMES_FLAT:
            note_idx = NOTE_NAMES_FLAT.index(note_part)
        else:
            note_idx = 0
        
        octave = int(octave_part) if octave_part else 4
        return (octave + 1) * 12 + note_idx
    except (ValueError, IndexError):
        return 60

def is_black_key(note_num: int) -> bool:
    return note_num % 12 in [1, 3, 6, 8, 10]

def apply_velocity_curve(value: float, curve_type: str) -> float:
    if curve_type == "Linear":
        return value
    elif curve_type == "Exponential":
        return value ** 2
    elif curve_type == "S-Curve":
        return (math.tanh((value - 0.5) * 4) + 1) / 2
    return value

def calculate_velocity(blob, mode: str, min_vel: int, max_vel: int, curve: str,
                      blend: float = 0.5, size_range: Tuple[int, int] = (300, 5000),
                      speed_range: Tuple[int, int] = (0, 50)) -> int:
    if mode == "Fixed":
        return max_vel

    size_norm = (blob.smooth_area - size_range[0]) / (size_range[1] - size_range[0])
    size_norm = max(0.0, min(1.0, size_norm))

    speed_norm = blob.smooth_speed / speed_range[1]
    speed_norm = max(0.0, min(1.0, speed_norm))

    if mode == "Size":
        raw_value = size_norm
    elif mode == "Speed":
        raw_value = speed_norm
    elif mode == "Size+Speed":
        raw_value = (size_norm * (1 - blend)) + (speed_norm * blend)
    else:
        raw_value = 1.0

    curved_value = apply_velocity_curve(raw_value, curve)
    velocity = int(min_vel + (curved_value * (max_vel - min_vel)))
    return max(1, min(127, velocity))

def calculate_modulation(blob, lfo_val: float, mod_type: str, depth: float) -> int:
    """Calculate modulation value from blob speed + LFO.

    Args:
        blob: BlobEntity with smooth_speed
        lfo_val: LFO sine wave value (-1.0 to +1.0)
        mod_type: "modwheel" or "pitchbend"
        depth: User-adjustable depth (0-127 for CC, 0.0-1.0 for pitchbend)

    Returns:
        int: MIDI value (0-127 for CC1, -8192 to +8191 for pitchbend)
    """
    # Normalize speed to 0-1 range (cap at 30 pixels/frame)
    speed_norm = min(1.0, blob.smooth_speed / 30.0)

    if mod_type == "modwheel":
        # Convert LFO from -1..1 to 0..1 (unipolar)
        lfo_norm = (lfo_val + 1.0) / 2.0

        # Blend: 30% speed + 70% LFO (LFO-dominant for smooth oscillation)
        modulation = (speed_norm * 0.3) + (lfo_norm * 0.7)

        # Scale by depth (0-127)
        cc1_value = int(modulation * depth)
        return max(0, min(127, cc1_value))

    elif mod_type == "pitchbend":
        # Keep LFO as -1..1 (bipolar) for pitch bend up/down
        lfo_bipolar = lfo_val

        # Blend: 20% speed + 80% LFO (LFO-dominant for organic pitch drift)
        # Speed adds subtle upward drift when moving
        modulation = (speed_norm * 0.2) + (lfo_bipolar * 0.8)

        # Scale by depth (0.0-1.0, where 1.0 = full ±8192 range)
        # depth=0.25 means ±2048, roughly ±2 semitones on most synths
        pitchbend_value = int(modulation * depth * 8192)
        return max(-8192, min(8191, pitchbend_value))

    return 0

def get_next_arp_note(arp_state: dict, mode: str, note_pool: List[int]) -> int:
    """Get next note in arpeggio sequence.

    Args:
        arp_state: Dict with 'index', 'direction', 'note_pool' keys
        mode: Arpeggiator mode (UP, DOWN, UP-DOWN, RANDOM)
        note_pool: List of available MIDI notes

    Returns:
        int: Next MIDI note number
    """
    if not note_pool:
        return 60  # Fallback to middle C

    if mode == "RANDOM":
        return random.choice(note_pool)

    if mode == "UP":
        note = note_pool[arp_state['index']]
        arp_state['index'] = (arp_state['index'] + 1) % len(note_pool)
        return note

    if mode == "DOWN":
        note = note_pool[arp_state['index']]
        arp_state['index'] = (arp_state['index'] - 1) % len(note_pool)
        return note

    if mode == "UP-DOWN":
        note = note_pool[arp_state['index']]

        # Calculate next index with ping-pong logic
        next_idx = arp_state['index'] + arp_state['direction']

        # Reverse direction at boundaries
        if next_idx >= len(note_pool):
            arp_state['direction'] = -1
            next_idx = len(note_pool) - 2 if len(note_pool) > 1 else 0
        elif next_idx < 0:
            arp_state['direction'] = 1
            next_idx = 1 if len(note_pool) > 1 else 0

        arp_state['index'] = max(0, min(len(note_pool) - 1, next_idx))
        return note

    return note_pool[0]  # Fallback

# ============================================================================
# DATA MODELS
# ============================================================================

class ChannelConfig:
    def __init__(self, config_id: int):
        self.config_id = config_id
        self.enabled = False
        self.name = f"Channel {config_id + 1}"
        self.midi_channel = config_id
        self.min_velocity = 40
        self.max_velocity = 100
        self.min_note = 36
        self.max_note = 84
        self.size_categories: set = {"All"}  # NEW: Multi-select size categories
        self.size_filter = "All"  # Keep for backward compatibility
        self.min_area = 0
        self.max_area = 10000
        self.color_filter_enabled = False
        self.color_hue_min = 0
        self.color_hue_max = 180
        self.color_sat_min = 0
        self.color_sat_max = 255
        self.probability = 100
        self.override_scale = None
        self.override_root = None
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
                 "#00BCD4", "#CDDC39", "#FF5722", "#3F51B5", "#009688",
                 "#FFC107", "#E91E63", "#8BC34A", "#673AB7", "#795548", "#607D8B"]
        self.display_color = colors[config_id % len(colors)]
        self.priority = config_id
        # Modulation settings (NEW)
        self.modwheel_enabled = True          # Enable CC1 modulation
        self.modwheel_depth = 60              # Depth: 0-127 (default: moderate)
        self.pitchbend_enabled = True         # Enable pitchbend modulation
        self.pitchbend_depth = 0.25           # Depth: 0.0-1.0 (0.25 = ±2 semitones)
        # Note duration settings
        self.min_note_duration = 0            # Minimum note duration in milliseconds (0 = no minimum)
        self.duration_deviation = 0           # Random deviation in milliseconds (±)

        # CC Modulation smoothing (laziness)
        self.cc_smoothing = 0.5               # Smoothing factor 0.0-1.0 (0=instant, 1=very lazy, default=50%)

        # Arpeggiator settings
        self.arp_mode = "OFF"                 # Arpeggiator mode: OFF, UP, DOWN, UP-DOWN, RANDOM

    def _matches_size_categories(self, blob) -> bool:
        """Check if blob's categories overlap with channel's selection (OR logic)."""
        if "All" in self.size_categories:
            return True
        if not self.size_categories:
            return False  # No categories selected = reject all
        # OR logic: blob matches if ANY of its categories are in channel's selection
        return bool(blob.size_categories & self.size_categories)

    def matches_blob(self, blob, frame_hsv=None) -> bool:
        if not self.enabled:
            return False

        # NEW: Size category check
        if not self._matches_size_categories(blob):
            return False

        # Color filter logic
        if self.color_filter_enabled and frame_hsv is not None:
            x, y = int(blob.smooth_x), int(blob.smooth_y)
            if 0 <= x < frame_hsv.shape[1] and 0 <= y < frame_hsv.shape[0]:
                h, s, v = frame_hsv[y, x]
                if not (self.color_hue_min <= h <= self.color_hue_max):
                    return False
                if not (self.color_sat_min <= s <= self.color_sat_max):
                    return False
        # Probability moved to process_logic to act as a trigger gate rather than a tracking gate
        return True

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "enabled": self.enabled,
            "name": self.name,
            "midi_channel": self.midi_channel,
            "min_velocity": self.min_velocity,
            "max_velocity": self.max_velocity,
            "min_note": self.min_note,
            "max_note": self.max_note,
            "size_categories": list(self.size_categories),  # NEW: Serialize as list
            "size_filter": self.size_filter,  # Keep for backward compatibility
            "min_area": self.min_area,
            "max_area": self.max_area,
            "color_filter_enabled": self.color_filter_enabled,
            "color_hue_min": self.color_hue_min,
            "color_hue_max": self.color_hue_max,
            "color_sat_min": self.color_sat_min,
            "color_sat_max": self.color_sat_max,
            "probability": self.probability,
            "override_scale": self.override_scale,
            "override_root": self.override_root,
            "display_color": self.display_color,
            "priority": self.priority,
            "modwheel_enabled": self.modwheel_enabled,
            "modwheel_depth": self.modwheel_depth,
            "pitchbend_enabled": self.pitchbend_enabled,
            "pitchbend_depth": self.pitchbend_depth,
            "min_note_duration": self.min_note_duration,
            "duration_deviation": self.duration_deviation,
            "cc_smoothing": self.cc_smoothing,
            "arp_mode": self.arp_mode
        }

    @staticmethod
    def from_dict(data: dict) -> 'ChannelConfig':
        config_id = data.get("config_id", data.get("channel", 0))
        ch = ChannelConfig(config_id)
        ch.enabled = data.get("enabled", False)
        ch.name = data.get("name", f"Channel {config_id + 1}")
        ch.midi_channel = data.get("midi_channel", data.get("channel", config_id))
        ch.min_velocity = data.get("min_velocity", 40)
        ch.max_velocity = data.get("max_velocity", 100)
        ch.min_note = data.get("min_note", 36)
        ch.max_note = data.get("max_note", 84)

        # NEW: Load size_categories with migration from old format
        if "size_categories" in data:
            ch.size_categories = set(data["size_categories"])
        else:
            # Backward compatibility: migrate old size_filter to new system
            old_filter = data.get("size_filter", "All")
            if old_filter == "All":
                ch.size_categories = {"All"}
            else:
                ch.size_categories = {old_filter}

        ch.size_filter = data.get("size_filter", "All")
        ch.min_area = data.get("min_area", 0)
        ch.max_area = data.get("max_area", 10000)
        ch.color_filter_enabled = data.get("color_filter_enabled", False)
        ch.color_hue_min = data.get("color_hue_min", 0)
        ch.color_hue_max = data.get("color_hue_max", 180)
        ch.color_sat_min = data.get("color_sat_min", 0)
        ch.color_sat_max = data.get("color_sat_max", 255)
        ch.probability = data.get("probability", 100)
        ch.override_scale = data.get("override_scale")
        ch.override_root = data.get("override_root")
        ch.display_color = data.get("display_color", "#FFFFFF")
        ch.priority = data.get("priority", config_id)
        ch.modwheel_enabled = data.get("modwheel_enabled", True)
        ch.modwheel_depth = data.get("modwheel_depth", 60)
        ch.pitchbend_enabled = data.get("pitchbend_enabled", True)
        ch.pitchbend_depth = data.get("pitchbend_depth", 0.25)
        ch.min_note_duration = data.get("min_note_duration", 0)
        ch.duration_deviation = data.get("duration_deviation", 0)
        ch.cc_smoothing = data.get("cc_smoothing", 0.5)
        ch.arp_mode = data.get("arp_mode", "OFF")
        return ch

class BlobEntity:
    def __init__(self, bid: int, x: float, y: float, area: float):
        self.id = bid
        self.x, self.y, self.area = x, y, area
        self.vx, self.vy = 0, 0
        self.smooth_x, self.smooth_y = x, y
        self.smooth_area = area
        self.smooth_speed = 0.0
        # Multi-channel assignment (locked on first ATTACK)
        self.channel_configs: List[ChannelConfig] = []  # All matching channels
        self.playing_notes: Dict[int, int] = {}  # {config_id: note_number}
        self.note_start_times: Dict[int, float] = {}  # {config_id: timestamp} for minimum duration enforcement
        self.smoothed_cc_values: Dict[int, Dict[int, float]] = {}  # {config_id: {cc_number: smoothed_value}}
        self.arp_states: Dict[int, Dict] = {}  # {config_id: {index, direction, note_pool}} for arpeggiator
        self.state = "OFF"
        self.state_timer = 0.0
        self.color_fade_timer = 0.0  # Timer for 20ms color fade
        self.lfo_phase = bid * (math.pi / 4)
        self.lfo_rate = random.uniform(0.5, 1.5)
        self.trail: List[Tuple[int, int]] = []
        self.collision_cooldown = 0.0
        self.size_categories: set = set()  # NEW: Track which size categories this blob belongs to

    def update(self, x: float, y: float, area: float, dt: float, smoothing: float = 0.1):
        self.vx = x - self.x
        self.vy = y - self.y
        raw_speed = math.hypot(self.vx, self.vy)
        self.x, self.y, self.area = x, y, area
        self.smooth_x += (x - self.smooth_x) * smoothing
        self.smooth_y += (y - self.smooth_y) * smoothing
        self.smooth_area += (area - self.smooth_area) * smoothing
        self.update_size_categories()  # NEW: Update categories after area changes
        self.smooth_speed += (raw_speed - self.smooth_speed) * smoothing
        self.trail.append((int(self.smooth_x), int(self.smooth_y)))
        if len(self.trail) > 20:
            self.trail.pop(0)
        if self.collision_cooldown > 0:
            self.collision_cooldown -= dt

    def update_size_categories(self):
        """Calculate which size categories this blob belongs to based on smooth_area."""
        self.size_categories.clear()
        for category_name, (min_area, max_area) in SIZE_PRESETS.items():
            if category_name == "All":
                continue  # Skip "All" as it's a UI convenience, not a real category
            if min_area <= self.smooth_area <= max_area:
                self.size_categories.add(category_name)

    def get_primary_size_category(self) -> str:
        """Get the most appropriate size category for display purposes.
        Priority: Tiny > Small > Medium > Large (smallest first)
        """
        priority_order = ["Tiny", "Small", "Medium", "Large"]
        for category in priority_order:
            if category in self.size_categories:
                return category
        return "Unknown"

# ============================================================================
# MIDI ENGINE
# ============================================================================

class MidiMonitor:
    def __init__(self):
        self.active_notes: Dict[int, Dict[int, Tuple[int, float]]] = defaultdict(dict)
        self.lock = Lock()
        self.cc_values: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.last_activity: Dict[int, float] = defaultdict(float)
        self.active_keys: Dict[int, float] = {} 
        self.note_channels: Dict[int, int] = {} 

    def note_on(self, channel: int, note: int, velocity: int):
        with self.lock:
            self.active_notes[channel][note] = (velocity, time.time())
            self.last_activity[channel] = time.time()
            self.active_keys[note] = time.time()
            self.note_channels[note] = channel 

    def note_off(self, channel: int, note: int):
        with self.lock:
            if channel in self.active_notes and note in self.active_notes[channel]:
                del self.active_notes[channel][note]
            if channel in self.active_notes and not self.active_notes[channel]:
                del self.active_notes[channel]
            if note in self.active_keys:
                del self.active_keys[note]

    def cc_update(self, channel: int, cc_num: int, value: int):
        with self.lock:
            self.cc_values[channel][cc_num] = value
            self.last_activity[channel] = time.time()

    def get_active_keys(self) -> Dict[int, float]:
        with self.lock:
            return dict(self.active_keys)

    def get_note_channels(self) -> Dict[int, int]:
        with self.lock:
            return dict(self.note_channels)

    def clear(self):
        with self.lock:
            self.active_notes.clear()
            self.cc_values.clear()
            self.last_activity.clear()
            self.active_keys.clear()
            self.note_channels.clear()

class MidiEngine:
    def __init__(self, logger, monitor: Optional[MidiMonitor] = None):
        self.outport: Optional[mido.ports.BaseOutput] = None
        self.logger = logger
        self.monitor = monitor or MidiMonitor()

    def get_ports(self) -> List[str]:
        try:
            return mido.get_output_names()
        except:
            return []

    def open_port(self, name: str):
        if self.outport:
            self.outport.close()
        try:
            self.outport = mido.open_output(name)
            self.logger(f"MIDI - {name}")
        except Exception as e:
            self.logger(f"MIDI Error: {e}")

    def send_cc(self, ch: int, cc: int, val: int):
        if self.outport:
            self.outport.send(mido.Message('control_change', channel=ch, control=cc, value=val))
            self.monitor.cc_update(ch, cc, val)

    def note_on(self, ch: int, note: int, vel: int):
        if self.outport:
            self.outport.send(mido.Message('note_on', channel=ch, note=note, velocity=vel))
            self.monitor.note_on(ch, note, vel)
            note_name = note_to_name(note)
            self.logger(f"🎵 Ch{ch+1:2d} | {note_name:4s} ({note:3d}) | Vel:{vel:3d}")

    def note_off(self, ch: int, note: int):
        if self.outport:
            self.outport.send(mido.Message('note_off', channel=ch, note=note))
            self.monitor.note_off(ch, note)

    def send_pitchbend(self, ch: int, value: int):
        """Send pitchbend message.
        Args:
            ch: MIDI channel (0-15)
            value: Pitchbend value (-8192 to +8191, 0 = center/no bend)
        """
        if self.outport:
            self.outport.send(mido.Message('pitchwheel', channel=ch, pitch=value))

    def panic(self):
        if self.outport:
            self.outport.panic()
            self.monitor.clear()
            self.logger("MIDI Panic - All notes off")

class PresetManager:
    def __init__(self, preset_dir: str = "./presets"):
        self.preset_dir = preset_dir
        self.current_preset: Optional[str] = None
        if not os.path.exists(preset_dir):
            os.makedirs(preset_dir)

    def list_presets(self) -> List[str]:
        if not os.path.exists(self.preset_dir):
            return []
        return [f for f in os.listdir(self.preset_dir) if f.endswith('.json')]

    def load_preset(self, filename: str) -> Optional[dict]:
        try:
            path = os.path.join(self.preset_dir, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                self.current_preset = filename
                return data
        except Exception as e:
            print(f"Error loading preset {filename}: {e}")
            return None

    def save_preset(self, filename: str, params: dict) -> bool:
        try:
            # Handle both absolute paths and simple filenames
            if not os.path.isabs(filename):
                path = os.path.join(self.preset_dir, filename)
            else:
                path = filename
                
            with open(path, 'w') as f:
                json.dump(params, f, indent=2)
            self.current_preset = filename
            return True
        except Exception as e:
            print(f"Error saving preset {filename}: {e}")
            return False

# ============================================================================
# VIRTUAL PIANO KEYBOARD
# ============================================================================

class VirtualPiano:
    def __init__(self, parent, note_range: Tuple[int, int] = (36, 96), height: int = 120, midi_engine=None, channel_configs=None):
        self.parent = parent
        self.note_range = note_range
        self.height = height
        self.active_notes: Dict[int, float] = {}
        self.key_states: Dict[int, float] = {}
        self.key_animations: Dict[int, float] = {}
        self.key_colors: Dict[int, str] = {}
        self.last_update = time.time()
        self.animation_duration = 0.15
        self.key_items: Dict[int, Dict[str, int]] = {}
        self.note_to_bounds: Dict[int, Tuple[float, float, float, float]] = {}
        self.piano_drawn = False
        self.midi_engine = midi_engine
        self.channel_configs = channel_configs or []
        self.clicked_notes: Dict[int, bool] = {}
        self.selected_channel = 0  # Default to channel 0 for manual testing

        # Main container with keyboard and controls
        self.frame = tk.Frame(parent, bg="#1a1a1a", height=height)
        self.frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        self.frame.pack_propagate(False)

        # Channel selector on the right
        control_frame = tk.Frame(self.frame, bg="#1a1a1a", width=80)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        control_frame.pack_propagate(False)

        tk.Label(control_frame, text="Channel", bg="#1a1a1a", fg="#ffffff",
                font=("Segoe UI", 8)).pack(pady=(5, 2))

        self.channel_var = tk.IntVar(value=0)
        channel_dropdown = tk.Spinbox(control_frame, from_=0, to=15,
                                     textvariable=self.channel_var, width=5,
                                     bg="#2a2a2a", fg="#ffffff",
                                     buttonbackground="#3a3a3a",
                                     insertbackground="#ffffff",
                                     relief=tk.FLAT, font=("Segoe UI", 9),
                                     command=self._on_channel_change)
        channel_dropdown.pack(pady=2)
        self.channel_var.trace_add("write", lambda *args: self._on_channel_change())

        # Keyboard canvas
        self.canvas = tk.Canvas(self.frame, bg="#1a1a1a", highlightthickness=0, height=height)
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.canvas.bind("<Configure>", lambda e: self._redraw_piano())
        self.canvas.bind("<Button-1>", self._on_key_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_key_release)
        self.canvas.bind("<Motion>", self._on_mouse_motion)

        self.frame.after(100, self._redraw_piano)
        
    def _redraw_piano(self):
        self.canvas.delete("all")
        self.key_items.clear()
        self.note_to_bounds.clear()
        self.canvas.update_idletasks()
        width = self.canvas.winfo_width()
        if width < 10:
            return

        min_note, max_note = self.note_range
        white_keys = []
        for note in range(min_note, max_note + 1):
            if not is_black_key(note):
                white_keys.append(note)

        white_key_width = width / len(white_keys) if white_keys else 1
        white_key_height = self.height - 10

        white_index = 0
        for note in range(min_note, max_note + 1):
            if not is_black_key(note):
                x = white_index * white_key_width
                brightness = self.key_states.get(note, 0.0)
                brightness = max(0.0, min(1.0, brightness))

                r = int(255 * brightness + 255 * (1 - brightness))
                g = int(68 * brightness + 255 * (1 - brightness))
                b = int(68 * brightness + 255 * (1 - brightness))
                fill_color = f"#{r:02x}{g:02x}{b:02x}"
                outline_color = "#ff0000" if brightness > 0.5 else "#cccccc"

                x1, y1 = x + 1, 5
                x2, y2 = x + white_key_width - 1, white_key_height
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill_color, outline=outline_color, width=2
                )
                self.note_to_bounds[note] = (x1, y1, x2, y2)

                note_name = NOTES[note % 12]
                octave = (note // 12) - 1
                label = f"{note_name}{octave}"
                text_color = "#fff" if brightness > 0.5 else "#333"
                text_id = self.canvas.create_text(
                    x + white_key_width / 2, white_key_height - 15,
                    text=label, fill=text_color,
                    font=("Arial", 7)
                )

                self.key_items[note] = {"rect": rect_id, "text": text_id}
                white_index += 1
        
        for note in range(min_note, max_note + 1):
            if is_black_key(note):
                white_before = sum(1 for n in range(min_note, note) if not is_black_key(n))
                x = white_before * white_key_width + white_key_width * 0.6

                brightness = self.key_states.get(note, 0.0)
                brightness = max(0.0, min(1.0, brightness))

                r = int(255 * brightness + 51 * (1 - brightness))
                g = int(0 * brightness + 51 * (1 - brightness))
                b = int(0 * brightness + 51 * (1 - brightness))
                fill_color = f"#{r:02x}{g:02x}{b:02x}"
                outline_color = "#ff6666" if brightness > 0.5 else "#000000"

                x1, y1 = x, 5
                x2, y2 = x + white_key_width * 0.6, white_key_height * 0.6
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill_color, outline=outline_color, width=1
                )
                self.note_to_bounds[note] = (x1, y1, x2, y2)
                self.key_items[note] = {"rect": rect_id, "text": None}

        self.piano_drawn = True
    
    def _update_key_color(self, note: int):
        if note not in self.key_items:
            return
        brightness = self.key_states.get(note, 0.0)
        brightness = max(0.0, min(1.0, brightness))
        is_black = is_black_key(note)
        channel_color = self.key_colors.get(note, "#FF4444")

        if channel_color.startswith("#"):
            channel_r = int(channel_color[1:3], 16)
            channel_g = int(channel_color[3:5], 16)
            channel_b = int(channel_color[5:7], 16)
        else:
            channel_r, channel_g, channel_b = 255, 68, 68

        if is_black:
            r = int(channel_r * brightness + 51 * (1 - brightness))
            g = int(channel_g * brightness + 51 * (1 - brightness))
            b = int(channel_b * brightness + 51 * (1 - brightness))
            fill_color = f"#{r:02x}{g:02x}{b:02x}"
            outline_r = min(255, int(channel_r * 1.3))
            outline_g = min(255, int(channel_g * 1.3))
            outline_b = min(255, int(channel_b * 1.3))
            outline_color = f"#{outline_r:02x}{outline_g:02x}{outline_b:02x}" if brightness > 0.5 else "#000000"
        else:
            r = int(channel_r * brightness + 255 * (1 - brightness))
            g = int(channel_g * brightness + 255 * (1 - brightness))
            b = int(channel_b * brightness + 255 * (1 - brightness))
            fill_color = f"#{r:02x}{g:02x}{b:02x}"
            outline_color = channel_color if brightness > 0.5 else "#cccccc"

        rect_id = self.key_items[note]["rect"]
        self.canvas.itemconfig(rect_id, fill=fill_color, outline=outline_color)

        if not is_black and self.key_items[note]["text"]:
            text_id = self.key_items[note]["text"]
            text_color = "#fff" if brightness > 0.5 else "#333"
            self.canvas.itemconfig(text_id, fill=text_color)
    
    def _update_animations(self, fade_states: dict = None):
        """Update key animations using exponential easing from fade_states.

        Args:
            fade_states: Optional dict of {note: fade_value} from App-level tracking
        """
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        changed_notes = set()

        # Use App-level fade states if provided and not empty, otherwise fall back to internal animation
        if fade_states is not None and len(fade_states) > 0:
            # Sync internal key_states with App-level fade_states
            for note, fade_value in fade_states.items():
                old_brightness = self.key_states.get(note, 0.0)
                self.key_states[note] = fade_value

                if abs(fade_value - old_brightness) > 0.01:
                    changed_notes.add(note)

            # Clean up keys that are no longer fading (fade back to 0)
            for note in list(self.key_states.keys()):
                if note not in fade_states:
                    del self.key_states[note]
                    changed_notes.add(note)
        elif fade_states is not None and len(fade_states) == 0:
            # Fade states exists but is empty - clear all key states
            for note in list(self.key_states.keys()):
                del self.key_states[note]
                changed_notes.add(note)
        else:
            # Original cubic easing animation (fallback)
            for note in list(self.key_animations.keys()):
                old_brightness = self.key_states.get(note, 0.0)
                if note in self.active_notes:
                    self.key_animations[note] += dt / self.animation_duration
                    if self.key_animations[note] > 1.0:
                        self.key_animations[note] = 1.0
                        self.key_states[note] = 1.0
                    else:
                        t = self.key_animations[note]
                        self.key_states[note] = 1 - (1 - t) ** 3
                else:
                    if note in self.key_states:
                        self.key_animations[note] -= dt / self.animation_duration
                        if self.key_animations[note] <= 0.0:
                            self.key_animations[note] = 0.0
                            self.key_states[note] = 0.0
                            del self.key_animations[note]
                            if note in self.key_states:
                                del self.key_states[note]
                        else:
                            t = self.key_animations[note]
                            self.key_states[note] = t ** 3

                new_brightness = self.key_states.get(note, 0.0)
                if abs(old_brightness - new_brightness) > 0.01:
                    changed_notes.add(note)
        
        for note in changed_notes:
            self._update_key_color(note)
    
    def update_active_notes(self, active_notes: Dict[int, float], note_channels: Dict[int, int] = None, fade_states: dict = None):
        """Update active notes and optionally apply smooth fade transitions.

        Args:
            active_notes: Dict of {note: velocity} for currently active notes
            note_channels: Optional dict of {note: channel_idx} for per-channel colors
            fade_states: Optional dict of {note: fade_value} for smooth transitions
        """
        if not self.piano_drawn:
            return
        if note_channels and self.channel_configs:
            for note, channel_idx in note_channels.items():
                if 0 <= channel_idx < len(self.channel_configs):
                    channel_config = self.channel_configs[channel_idx]
                    self.key_colors[note] = channel_config.display_color

        for note in active_notes:
            if note not in self.key_animations:
                self.key_animations[note] = 0.0
                self.key_states[note] = 0.0

        for note in list(self.key_colors.keys()):
            if note not in self.key_animations and note not in active_notes:
                del self.key_colors[note]

        self.active_notes = active_notes
        self._update_animations(fade_states)
    
    def set_note_range(self, min_note: int, max_note: int):
        self.note_range = (min_note, max_note)
        self._redraw_piano()

    def _find_note_at_position(self, x: int, y: int) -> Optional[int]:
        for note, bounds in self.note_to_bounds.items():
            if is_black_key(note):
                x1, y1, x2, y2 = bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return note
        for note, bounds in self.note_to_bounds.items():
            if not is_black_key(note):
                x1, y1, x2, y2 = bounds
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return note
        return None

    def _on_channel_change(self):
        """Update selected channel when dropdown changes."""
        self.selected_channel = self.channel_var.get()

    def _on_key_press(self, event):
        note = self._find_note_at_position(event.x, event.y)
        if note is not None and self.midi_engine is not None:
            if note not in self.clicked_notes or not self.clicked_notes[note]:
                self.clicked_notes[note] = True
                self.midi_engine.note_on(self.selected_channel, note, 100)
                self.active_notes[note] = time.time()

    def _on_key_release(self, event):
        for note in list(self.clicked_notes.keys()):
            if self.clicked_notes[note] and self.midi_engine is not None:
                self.midi_engine.note_off(self.selected_channel, note)
                self.clicked_notes[note] = False

    def _on_mouse_motion(self, event):
        if event.state & 0x0100:
            note = self._find_note_at_position(event.x, event.y)
            for clicked_note in list(self.clicked_notes.keys()):
                if self.clicked_notes[clicked_note] and clicked_note != note:
                    if self.midi_engine is not None:
                        self.midi_engine.note_off(self.selected_channel, clicked_note)
                    self.clicked_notes[clicked_note] = False
            if note is not None and (note not in self.clicked_notes or not self.clicked_notes[note]):
                if self.midi_engine is not None:
                    self.clicked_notes[note] = True
                    self.midi_engine.note_on(self.selected_channel, note, 100)
                    self.active_notes[note] = time.time()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class App:
    """Main application class"""
    def __init__(self, window: tk.Tk, title: str):
        self.window = window
        self.window.title(title)
        self.window.geometry("1920x1080")
        self.window.configure(bg="#1a1a1a")
        try:
            self.window.state('zoomed')
        except:
            pass

        # Initialize components
        self.midi_monitor = MidiMonitor()
        self.midi = MidiEngine(self.log, self.midi_monitor)
        self.preset_manager = PresetManager()
        
        # Initial variable setup (will be overwritten by UI creators, but safe defaults)
        self.bg_threshold_var = tk.IntVar(value=16) 
        self.bg_history_var = tk.IntVar(value=500)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.bg_history_var.get(), 
                                                       varThreshold=self.bg_threshold_var.get(), 
                                                       detectShadows=False)

        # State
        self.blobs: Dict[int, BlobEntity] = {}
        self.next_id = 0
        self.midi_active = False
        self.active_scale_notes: List[int] = []
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

        # Keyboard fade states for smooth transitions
        self.key_fade_states = {}  # {note_number: fade_value (0.0-1.0)}

        # Channel configurations
        self.channel_configs = [ChannelConfig(i) for i in range(16)]
        self.active_channel_uis: List[Dict] = []

        # NEW: Ensure all configs have size_categories (migration from old code)
        for cfg in self.channel_configs:
            if not hasattr(cfg, 'size_categories') or not cfg.size_categories:
                if hasattr(cfg, 'size_filter') and cfg.size_filter in SIZE_PRESETS:
                    cfg.size_categories = {cfg.size_filter}
                else:
                    cfg.size_categories = {"All"}

        # UI Setup
        self.log_box = None  # Initialize early, will be created in _create_piano_keyboard
        self._setup_ui()
        self.piano = None
        self.update_scale_pool()
        self._try_load_default_preset()
        self.update_loop()
        self.update_midi_monitor()
        self.window.after(200, self._create_piano_keyboard)
        self.window.after(300, self.update_piano)

    # ========================================================================
    # UI SETUP
    # ========================================================================

    def _setup_ui(self):
        """Setup GUI matching screenshot style: Left sidebar, Center video, Right sidebar"""
        main_container = tk.Frame(self.window, bg="#1a1a1a")
        main_container.pack(fill=tk.BOTH, expand=True)

        main_paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, sashwidth=6,
                                   bg="#0d0d0d", sashrelief=tk.RAISED, bd=0)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        screen_width = self.window.winfo_screenwidth()
        # Set left column to 1/5th of screen width + 20px (increased from 1/6 then 1/5)
        left_width = screen_width // 5 + 20
        center_width = int(screen_width * 0.5)
        right_width = screen_width - left_width - center_width

        # === LEFT SIDEBAR ===
        left_sidebar = tk.Frame(main_paned, bg="#1e1e1e", width=left_width)
        main_paned.add(left_sidebar, minsize=200, width=left_width)
        self._create_left_sidebar(left_sidebar)

        # === CENTER: Video Area ===
        center_panel = tk.Frame(main_paned, bg="#000", width=center_width)
        main_paned.add(center_panel, minsize=400, width=center_width)
        self._create_video_panel(center_panel)

        # === RIGHT SIDEBAR ===
        right_sidebar = tk.Frame(main_paned, bg="#1e1e1e", width=right_width)
        main_paned.add(right_sidebar, minsize=300, width=right_width)
        self._create_right_sidebar(right_sidebar)

        # === BOTTOM: Status Bar ===
        status_bar = tk.Frame(main_container, bg="#2b2b2b", height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, before=None)
        status_bar.pack_propagate(False)

        self.status_midi = tk.Label(status_bar, text="MIDI: Ready ○", bg="#2b2b2b", fg="#888",
                                    font=("Arial", 10, "bold"))
        self.status_midi.pack(side=tk.LEFT, padx=15, pady=5)
        self.status_fps = tk.Label(status_bar, text="0 FPS", bg="#2b2b2b", fg="#888",
                                   font=("Arial", 10))
        self.status_fps.pack(side=tk.LEFT, padx=15, pady=5)
        self.status_thread = tk.Label(status_bar, text="System Ready ✓", bg="#2b2b2b", fg="#4CAF50",
                                      font=("Arial", 10))
        self.status_thread.pack(side=tk.RIGHT, padx=15, pady=5)

    def _bind_mouse_scroll(self, widget, canvas):
        """Helper to bind mouse wheel to a canvas when hovering a widget"""
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Windows/MacOS MouseWheel
        widget.bind("<Enter>", lambda _: widget.bind_all("<MouseWheel>", _on_mousewheel))
        widget.bind("<Leave>", lambda _: widget.unbind_all("<MouseWheel>"))
        # Linux Button-4/5
        widget.bind("<Button-4>", lambda _: canvas.yview_scroll(-1, "units"))
        widget.bind("<Button-5>", lambda _: canvas.yview_scroll(1, "units"))

    def _bind_mouse_scroll_recursive(self, widget, canvas):
        """Recursively bind mouse scroll to widget and all children."""
        self._bind_mouse_scroll(widget, canvas)

        # Bind to all children recursively
        for child in widget.winfo_children():
            self._bind_mouse_scroll_recursive(child, canvas)

    def _create_param_slider(self, parent, min_val, max_val, default_val, resolution, var_type=tk.IntVar, callback=None, indicator=None):
        """
        Creates a consistent slider with a manual entry box next to it.
        Returns the variable associated with the slider.
        """
        container = tk.Frame(parent, bg="#1e1e1e")
        container.pack(fill=tk.X, expand=True)

        frame = tk.Frame(container, bg="#1e1e1e")
        frame.pack(fill=tk.X, expand=True)

        var = var_type(value=default_val)
        if callback:
            var.trace("w", lambda *a: callback(var.get()))

        # Slider
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var,
                          resolution=resolution, bg="#1e1e1e", fg="white", highlightthickness=0,
                          showvalue=False, length=100) # Hide internal value, show via entry
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Reset on double click
        slider.bind("<Double-Button-1>", lambda e: var.set(default_val))

        # Entry/Spinbox
        entry = tk.Spinbox(frame, from_=min_val, to=max_val, increment=resolution,
                           textvariable=var, width=5, bg="#2b2b2b", fg="white",
                           relief=tk.FLAT, buttonbackground="#2b2b2b")
        entry.pack(side=tk.RIGHT)

        # Optional Indicator (now below the slider)
        if indicator:
             tk.Label(container, text=indicator, bg="#1e1e1e", fg="#888", font=("Arial", 7)).pack(anchor="w", padx=2)

        return var

    def _create_left_sidebar(self, parent):
        header = tk.Frame(parent, bg="#2b2b2b", height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="MAGMA", font=("Arial", 16, "bold"), bg="#2b2b2b", fg="#9C27B0").pack(side=tk.LEFT, padx=15, pady=12)
        tk.Label(header, text="v7.6", font=("Arial", 10), bg="#2b2b2b", fg="#888").pack(side=tk.LEFT, pady=12)

        canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg="#1e1e1e")

        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind scroll recursively to all widgets
        self._bind_mouse_scroll_recursive(scrollable, canvas)
        self._bind_mouse_scroll(parent, canvas)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._create_collapsible_section(scrollable, "Project", [
            ("Save Preset", self._create_save_preset_button),
            ("Load Preset", self._create_load_preset_button),
        ], start_collapsed=False)

        self._create_collapsible_section(scrollable, "Video", [
            ("Source", self._create_video_source_controls),
            ("Brightness", self._create_brightness_control),
            ("Contrast", self._create_contrast_control),
            ("Overlays", self._create_overlay_toggles),
        ], start_collapsed=False)

        self._create_collapsible_section(scrollable, "Music", [
            ("Scale", self._create_scale_controls),
            ("Root", self._create_root_control),
            ("Chord Mask", self._create_chord_mask_control),
        ], start_collapsed=True)

        self._create_collapsible_section(scrollable, "Motion Detection", [
            ("Sensitivty (Thresh)", self._create_motion_threshold_control),
            ("BG History", self._create_motion_history_control),
            ("Trigger Speed", self._create_sensitivity_control),
            ("Blob Border Colors", self._create_blob_colors_control),
        ], start_collapsed=True)

        self._create_collapsible_section(scrollable, "Velocity", [
            ("Mode", self._create_velocity_mode_control),
            ("Min/Max", self._create_velocity_range_control),
            ("Curve", self._create_velocity_curve_control),
        ], start_collapsed=True)

        self._create_collapsible_section(scrollable, "Output", [
            ("MIDI Port", self._create_midi_port_control),
        ], start_collapsed=True)

    def _create_collapsible_section(self, parent, title, items, start_collapsed=False):
        section_frame = tk.Frame(parent, bg="#1e1e1e")
        section_frame.pack(fill=tk.X, padx=5, pady=2)
        header = tk.Frame(section_frame, bg="#2b2b2b", cursor="hand2")
        header.pack(fill=tk.X)
        collapsed = tk.BooleanVar(value=start_collapsed)
        def toggle():
            if collapsed.get():
                content.pack(fill=tk.X, padx=5, pady=5)
                arrow.config(text="▼")
                collapsed.set(False)
            else:
                content.pack_forget()
                arrow.config(text="▶")
                collapsed.set(True)
        header.bind("<Button-1>", lambda e: toggle())
        arrow = tk.Label(header, text="▼" if not start_collapsed else "▶", bg="#2b2b2b", fg="white", font=("Arial", 8), width=2)
        arrow.pack(side=tk.LEFT, padx=5)
        arrow.bind("<Button-1>", lambda e: toggle())
        tk.Label(header, text=title, bg="#2b2b2b", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        content = tk.Frame(section_frame, bg="#1e1e1e")
        if not start_collapsed:
            content.pack(fill=tk.X, padx=5, pady=5)
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                label, creator = item
                if callable(creator):
                    creator(content)
                else:
                    btn = tk.Button(content, text=label, bg=creator, fg="white",
                                   font=("Arial", 9), command=creator if callable(creator) else None)
                    btn.pack(fill=tk.X, pady=2)
            elif callable(item):
                item(content)

    def _create_save_preset_button(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        btn = tk.Button(frame, text="💾 Save Preset", bg="#2196F3", fg="white",
                       font=("Arial", 9, "bold"), command=self.save_preset,
                       relief=tk.FLAT, cursor="hand2", activebackground="#1976D2")
        btn.pack(fill=tk.X, padx=5, pady=3)

    def _create_load_preset_button(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        btn = tk.Button(frame, text="📂 Load Preset", bg="#2196F3", fg="white",
                       font=("Arial", 9, "bold"), command=self.load_preset,
                       relief=tk.FLAT, cursor="hand2", activebackground="#1976D2")
        btn.pack(fill=tk.X, padx=5, pady=3)

    def _create_video_source_controls(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        self.video_source_var = tk.StringVar(value="Webcam")
        self.btn_webcam = tk.Button(frame, text="Webcam", bg="#2196F3", fg="white",
                                    font=("Arial", 9, "bold"), relief=tk.FLAT,
                                    command=lambda: self.set_src_button("cam"))
        self.btn_webcam.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.btn_file = tk.Button(frame, text="File", bg="#2b2b2b", fg="white",
                                  font=("Arial", 9), relief=tk.FLAT,
                                  command=lambda: self.set_src_button("file"))
        self.btn_file.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _create_brightness_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Brightness:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.var_brightness = self._create_param_slider(frame, -100, 100, 0, 1, indicator="(High=Bright)")

    def _create_contrast_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Contrast:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.var_contrast = self._create_param_slider(frame, 0, 200, 100, 1, indicator="(High=Vivid)")

    def _create_overlay_toggles(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        self.notes_toggle = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text="Show Notes", variable=self.notes_toggle, bg="#1e1e1e", fg="white",
                      selectcolor="#2b2b2b", activebackground="#1e1e1e").pack(side=tk.LEFT, padx=5)

    def _create_scale_controls(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Scale:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="Minor Pentatonic")
        cb = ttk.Combobox(frame, textvariable=self.scale_var, values=list(SCALES.keys()),
                         state="readonly")
        cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.update_scale_pool())
        self.scale_var.trace("w", lambda *a: self.update_scale_pool())

    def _create_root_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Root:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.root_var = tk.StringVar(value="C")
        cb = ttk.Combobox(frame, textvariable=self.root_var, values=NOTES, state="readonly")
        cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.root_var.trace("w", lambda *a: self.update_scale_pool())

    def _create_chord_mask_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Mask:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.chord_var = tk.StringVar(value="Full Scale")
        cb = ttk.Combobox(frame, textvariable=self.chord_var, values=list(CHORD_MASKS.keys()),
                         state="readonly")
        cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.update_scale_pool())

    def _create_motion_threshold_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Det. Thresh:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.bg_threshold_var = self._create_param_slider(frame, 1, 100, 16, 1, tk.IntVar, indicator="(Low=Sensitive)")

    def _create_motion_history_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="BG History:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.bg_history_var = self._create_param_slider(frame, 10, 1000, 500, 10, tk.IntVar, indicator="(High=Stable)")

    def _create_sensitivity_control(self, parent):
        frame1 = tk.Frame(parent, bg="#1e1e1e")
        frame1.pack(fill=tk.X, pady=2)
        tk.Label(frame1, text="Min Size:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.var_size = self._create_param_slider(frame1, 5, 5000, 300, 5, indicator="(Low=Tiny)")
        
        frame2 = tk.Frame(parent, bg="#1e1e1e")
        frame2.pack(fill=tk.X, pady=2)
        tk.Label(frame2, text="Trigger:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.var_trigger = self._create_param_slider(frame2, 0.0, 50.0, 2.0, 0.1, tk.DoubleVar, indicator="(Low=Sensitive)")

    def _create_blob_colors_control(self, parent):
        """Create color pickers for blob border states."""
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)

        # Initialize color variables (BGR format for OpenCV)
        if not hasattr(self, 'blob_color_off'):
            self.blob_color_off = (0, 255, 0)      # Green - OFF state
            self.blob_color_attack = (0, 165, 255) # Orange - ATTACK state
            self.blob_color_on = (0, 0, 255)       # Red - ON state

        # Create color buttons for each state
        tk.Label(frame, text="State Colors:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)

        states_frame = tk.Frame(frame, bg="#1e1e1e")
        states_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # OFF state (green)
        self._create_color_button(states_frame, "OFF", self.blob_color_off,
                                  lambda c: setattr(self, 'blob_color_off', c))

        # ATTACK state (orange)
        self._create_color_button(states_frame, "ATTACK", self.blob_color_attack,
                                  lambda c: setattr(self, 'blob_color_attack', c))

        # ON state (red)
        self._create_color_button(states_frame, "ON", self.blob_color_on,
                                  lambda c: setattr(self, 'blob_color_on', c))

    def _create_unassigned_color_control(self, parent):
        """Create color picker for unassigned blob dots."""
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)

        # Initialize unassigned dot color (BGR format)
        if not hasattr(self, 'blob_unassigned_color'):
            self.blob_unassigned_color = (60, 60, 60)  # Dark gray

        tk.Label(frame, text="Dot Color:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)

        self._create_color_button(frame, "Color", self.blob_unassigned_color,
                                 lambda c: setattr(self, 'blob_unassigned_color', c))

    def _create_color_button(self, parent, label, current_color_bgr, on_change_callback):
        """Helper to create a color picker button."""
        # Convert BGR to RGB for display
        r, g, b = current_color_bgr[2], current_color_bgr[1], current_color_bgr[0]
        color_hex = f"#{r:02x}{g:02x}{b:02x}"

        btn = tk.Button(parent, text=label, bg=color_hex, fg="white" if (r+g+b) < 384 else "black",
                       font=("Arial", 8), width=8, relief=tk.FLAT, padx=5, pady=2)
        btn.pack(side=tk.LEFT, padx=2)

        def pick_color():
            from tkinter import colorchooser
            # Show color picker (returns RGB tuple and hex)
            color = colorchooser.askcolor(color=color_hex, title=f"Choose {label} Color")
            if color[0]:  # User selected a color
                rgb = color[0]
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                new_hex = f"#{r:02x}{g:02x}{b:02x}"
                # Convert RGB to BGR for OpenCV
                bgr = (b, g, r)
                # Update button appearance
                btn.config(bg=new_hex, fg="white" if (r+g+b) < 384 else "black")
                # Call the callback with BGR color
                on_change_callback(bgr)

        btn.config(command=pick_color)
        return btn

    def _create_velocity_mode_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Mode:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.vel_mode_var = tk.StringVar(value="Fixed")
        cb = ttk.Combobox(frame, textvariable=self.vel_mode_var,
                         values=["Fixed", "Size", "Speed", "Size+Speed"], state="readonly")
        cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _create_velocity_range_control(self, parent):
        frame1 = tk.Frame(parent, bg="#1e1e1e")
        frame1.pack(fill=tk.X, pady=2)
        tk.Label(frame1, text="Min:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.var_vel_min = self._create_param_slider(frame1, 1, 127, 40, 1, indicator="(High=Loud)")
        
        frame2 = tk.Frame(parent, bg="#1e1e1e")
        frame2.pack(fill=tk.X, pady=2)
        tk.Label(frame2, text="Max:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.var_vel_max = self._create_param_slider(frame2, 1, 127, 100, 1, indicator="(High=Loud)")

    def _create_velocity_curve_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Curve:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.vel_curve_var = tk.StringVar(value="Linear")
        cb = ttk.Combobox(frame, textvariable=self.vel_curve_var,
                         values=["Linear", "Exponential", "S-Curve"], state="readonly")
        cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _create_midi_port_control(self, parent):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Port:", bg="#1e1e1e", fg="white", width=10, anchor="w").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        ports = self.midi.get_ports() or ["No MIDI"]
        cb = ttk.Combobox(frame, textvariable=self.port_var, values=ports, state="readonly")
        cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        if ports:
            cb.current(0)
        cb.bind("<<ComboboxSelected>>", lambda e: self.midi.open_port(self.port_var.get()))
        refresh_btn = tk.Button(frame, text="🔄", font=("Arial", 8), bg="#2b2b2b", fg="white",
                               command=lambda: self._refresh_midi_ports(cb))
        refresh_btn.pack(side=tk.LEFT, padx=2)

    def _refresh_midi_ports(self, combobox):
        ports = self.midi.get_ports() or ["No MIDI"]
        combobox['values'] = ports
        if ports and ports[0] != "No MIDI":
            combobox.current(0)

    def _create_video_panel(self, parent):
        self.canvas = tk.Canvas(parent, bg="#000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas_width = 640
        self.canvas_height = 480

    def _on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height

    def _create_right_sidebar(self, parent):
        # Remove tabs - show channels directly with log below
        channels_container = tk.Frame(parent, bg="#1e1e1e")
        channels_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_channels_tab(channels_container)

    def _create_channels_tab(self, parent):
        header = tk.Frame(parent, bg="#2b2b2b", height=45)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="Channel Configuration", font=("Arial", 12, "bold"),
                bg="#2b2b2b", fg="white").pack(side=tk.LEFT, padx=15, pady=12)

        btn_add = tk.Button(header, text="+ Add Channel", bg="#4CAF50", fg="white",
                           font=("Arial", 9, "bold"), command=self.add_channel_ui,
                           relief=tk.FLAT, cursor="hand2", activebackground="#45a049")
        btn_add.pack(side=tk.RIGHT, padx=15, pady=8)

        btn_collapse = tk.Button(header, text="Collapse All", bg="#555555", fg="white",
                                font=("Arial", 9), command=self.collapse_all_channels,
                                relief=tk.FLAT, cursor="hand2", activebackground="#666666")
        btn_collapse.pack(side=tk.RIGHT, padx=5, pady=8)

        canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.channels_frame = tk.Frame(canvas, bg="#1e1e1e")

        self.channels_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self.channels_frame, anchor="nw")
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind scroll recursively to all widgets
        self._bind_mouse_scroll_recursive(self.channels_frame, canvas)
        self._bind_mouse_scroll(parent, canvas)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.active_channel_uis = []
        self.eyedropper_active = False

    def _create_log_content(self, parent):
        """Create compact log display (used next to keyboard)."""
        # Compact header
        header = tk.Frame(parent, bg="#2b2b2b")
        header.pack(fill=tk.X)
        tk.Label(header, text="Log", font=("Arial", 9, "bold"),
                bg="#2b2b2b", fg="white").pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(header, text="Clear", bg="#f44336", fg="white", font=("Arial", 7),
                 command=lambda: self.log_box.delete(1.0, tk.END), relief=tk.FLAT,
                 cursor="hand2", activebackground="#e53935", padx=3, pady=1).pack(side=tk.RIGHT, padx=5, pady=3)

        self.log_box = scrolledtext.ScrolledText(parent, height=8, font=("Consolas", 8),
                                                bg="#1a1a1a", fg="#00ff00", insertbackground="white",
                                                wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ========================================================================
    # VIRTUAL PIANO
    # ========================================================================

    def _create_piano_keyboard(self):
        if self.piano is not None:
            return
        piano_container = tk.Frame(self.window, bg="#1a1a1a", height=130)
        piano_container.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        piano_container.pack_propagate(False)

        button_frame = tk.Frame(piano_container, bg="#1a1a1a", width=150)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 10))
        button_frame.pack_propagate(False)

        start_btn = tk.Button(button_frame, text="▶ START\nMIDI", bg="#4CAF50", fg="white",
                             font=("Arial", 12, "bold"), command=self.toggle_midi,
                             relief=tk.FLAT, cursor="hand2", activebackground="#45a049",
                             wraplength=120)
        start_btn.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.btn_run = start_btn

        piano_frame = tk.Frame(piano_container, bg="#1a1a1a")
        piano_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.piano = VirtualPiano(piano_frame, note_range=(36, 96), height=120,
                                   midi_engine=self.midi, channel_configs=self.channel_configs)

        # Log panel to the right of keyboard
        log_frame = tk.Frame(piano_container, bg="#1a1a1a", width=300)
        log_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))
        log_frame.pack_propagate(False)
        self._create_log_content(log_frame)

    def update_piano(self):
        if self.piano is not None and self.piano.piano_drawn:
            active_keys = self.midi_monitor.get_active_keys()
            note_channels = self.midi_monitor.get_note_channels()
            # Always pass fade_states for smooth keyboard transitions
            self.piano.update_active_notes(active_keys, note_channels, self.key_fade_states)
        self.window.after(33, self.update_piano)

    def update_key_fades(self, dt: float, fade_speed: float = 5.0):
        """Update keyboard fade states for smooth transitions.

        Args:
            dt: Delta time in seconds
            fade_speed: Speed of fade transition (higher = faster)
        """
        active_notes = set(self.midi_monitor.get_active_keys().keys())

        for note in list(self.key_fade_states.keys()):
            target = 1.0 if note in active_notes else 0.0
            current = self.key_fade_states[note]

            # Exponential ease towards target
            diff = target - current
            self.key_fade_states[note] += diff * fade_speed * dt

            # Clean up fully faded out keys
            if self.key_fade_states[note] < 0.01 and target == 0.0:
                del self.key_fade_states[note]

        # Add newly active notes if not already tracked
        for note in active_notes:
            if note not in self.key_fade_states:
                self.key_fade_states[note] = 0.0  # Start from 0, fade in

    # ========================================================================
    # CHANNEL UI
    # ========================================================================

    def add_channel_ui(self, target_idx=None):
        """Add channel UI. If target_idx provided, use that, else find first empty."""
        channel_idx = target_idx
        
        if channel_idx is None:
            for i, cfg in enumerate(self.channel_configs):
                if not cfg.enabled:
                    channel_idx = i
                    cfg.enabled = True
                    break

        if channel_idx is None:
            self.log("All 16 channel slots in use!")
            return

        cfg = self.channel_configs[channel_idx]
        ch_frame = tk.Frame(self.channels_frame, bg="#1e1e1e", relief=tk.RAISED, borderwidth=1)
        ch_frame.pack(fill=tk.X, padx=5, pady=8)

        # Collapsible header
        header = tk.Frame(ch_frame, bg=cfg.display_color, cursor="hand2")
        header.pack(fill=tk.X)
        
        collapsed = tk.BooleanVar(value=False)
        def toggle_section():
            if collapsed.get():
                content.pack(fill=tk.X, padx=10, pady=10)
                arrow.config(text="▼")
                collapsed.set(False)
            else:
                content.pack_forget()
                arrow.config(text="▶")
                collapsed.set(True)
        
        header.bind("<Button-1>", lambda e: toggle_section())
        arrow = tk.Label(header, text="▼", bg=cfg.display_color, fg="white", font=("Arial", 10), width=2)
        arrow.pack(side=tk.LEFT, padx=5, pady=5)
        arrow.bind("<Button-1>", lambda e: toggle_section())

        # Channel name with MIDI channel number
        name_entry = tk.Entry(header, bg=cfg.display_color, fg="white", font=("Arial", 10, "bold"),
                             insertbackground="white", relief=tk.FLAT, borderwidth=0)
        name_entry.insert(0, cfg.name)
        name_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # MIDI channel indicator (display 1-16)
        midi_ch_label = tk.Label(header, text=f"Ch{cfg.midi_channel + 1}", bg=cfg.display_color,
                                fg="white", font=("Arial", 9), padx=8)
        midi_ch_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        del_btn = tk.Button(header, text="✕", bg="#f44336", fg="white", font=("Arial", 9),
                           command=lambda: self.remove_channel_ui(channel_idx), relief=tk.FLAT)
        del_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        content = tk.Frame(ch_frame, bg="#2b2b2b")
        content.pack(fill=tk.X, padx=10, pady=10)

        self._create_channel_section(content, "Basic", [
            ("MIDI Channel", lambda p: self._create_midi_ch_control(p, channel_idx, cfg)),
            ("Size Filter", lambda p: self._create_size_filter_control(p, channel_idx, cfg)),
            ("Probability", lambda p: self._create_probability_control(p, channel_idx, cfg)),
        ])

        self._create_channel_section(content, "Note Range", [
            ("Min Note", lambda p: self._create_note_min_control(p, channel_idx, cfg)),
            ("Max Note", lambda p: self._create_note_max_control(p, channel_idx, cfg)),
        ])

        self._create_channel_section(content, "Velocity", [
            ("Min/Max", lambda p: self._create_velocity_control(p, channel_idx, cfg)),
        ])

        self._create_channel_section(content, "Note Duration", [
            ("Min Duration (ms)", lambda p: self._create_min_duration_control(p, channel_idx, cfg)),
            ("Deviation (ms)", lambda p: self._create_duration_deviation_control(p, channel_idx, cfg)),
        ])

        self._create_channel_section(content, "Color Filter", [
            ("Color", lambda p: self._create_color_filter_control(p, channel_idx, cfg)),
        ])

        self._create_channel_section(content, "Modulation", [
            ("Modwheel (CC1)", lambda p: self._create_modwheel_control(p, channel_idx, cfg)),
            ("Pitchbend", lambda p: self._create_pitchbend_control(p, channel_idx, cfg)),
            ("CC Smoothing", lambda p: self._create_cc_smoothing_control(p, channel_idx, cfg)),
        ])

        ui_widgets = {
            "name_entry": name_entry,
            "container": ch_frame,
            "content": content,
            "collapsed": collapsed,
            "arrow": arrow,
            "midi_ch_label": midi_ch_label
        }
        name_entry.bind("<KeyRelease>", lambda e: self.update_channel_name(channel_idx, name_entry.get()))
        self.active_channel_uis.append({"index": channel_idx, "widgets": ui_widgets})
        if target_idx is None:
            self.log(f"Added {cfg.name}")

    def collapse_all_channels(self):
        """Collapse all channel configuration panels."""
        for ui_data in self.active_channel_uis:
            widgets = ui_data["widgets"]
            # Collapse if not already collapsed
            if not widgets["collapsed"].get():
                widgets["content"].pack_forget()
                widgets["arrow"].config(text="▶")
                widgets["collapsed"].set(True)

    def _create_channel_section(self, parent, title, items):
        section_frame = tk.Frame(parent, bg="#2b2b2b")
        section_frame.pack(fill=tk.X, pady=3)
        section_collapsed = tk.BooleanVar(value=False)
        def toggle():
            if section_collapsed.get():
                section_content.pack(fill=tk.X, padx=5, pady=3)
                section_arrow.config(text="▼")
                section_collapsed.set(False)
            else:
                section_content.pack_forget()
                section_arrow.config(text="▶")
                section_collapsed.set(True)
        section_header = tk.Frame(section_frame, bg="#1e1e1e", cursor="hand2")
        section_header.pack(fill=tk.X)
        section_header.bind("<Button-1>", lambda e: toggle())
        section_arrow = tk.Label(section_header, text="▼", bg="#1e1e1e", fg="white", font=("Arial", 8), width=2)
        section_arrow.pack(side=tk.LEFT, padx=3)
        section_arrow.bind("<Button-1>", lambda e: toggle())
        tk.Label(section_header, text=title, bg="#1e1e1e", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=3)
        section_content = tk.Frame(section_frame, bg="#2b2b2b")
        section_content.pack(fill=tk.X, padx=5, pady=3)
        for label, creator in items:
            creator(section_content)

    def _create_midi_ch_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="MIDI Ch:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)

        vals = [str(i+1) for i in range(16)]
        def on_change(event):
            try:
                val = int(cb.get()) - 1
                self.update_channel_midi(channel_idx, val)
            except ValueError:
                pass

        cb = ttk.Combobox(frame, values=vals, state="readonly", width=5)
        cb.set(str(cfg.midi_channel + 1))
        cb.bind("<<ComboboxSelected>>", on_change)
        cb.pack(side=tk.LEFT, padx=5)

    def _create_size_filter_control(self, parent, channel_idx, cfg):
        """Create checkbox grid for multi-category size selection."""
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)

        tk.Label(frame, text="Size Categories:", bg="#2b2b2b", fg="white",
                 width=15, anchor="w").pack(side=tk.LEFT, anchor="n", pady=5)

        checkbox_container = tk.Frame(frame, bg="#2b2b2b")
        checkbox_container.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        # Store checkbox variables
        checkbox_vars = {}

        # "All" checkbox (special handling)
        all_var = tk.BooleanVar(value="All" in cfg.size_categories)
        checkbox_vars["All"] = all_var
        all_cb = tk.Checkbutton(
            checkbox_container, text="All", variable=all_var,
            bg="#2b2b2b", fg="#FFD700", selectcolor="#1e1e1e",
            font=("Arial", 9, "bold"),
            command=lambda: self._toggle_size_category(channel_idx, "All", all_var, checkbox_vars)
        )
        all_cb.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        # Individual category checkboxes
        categories = ["Tiny", "Small", "Medium", "Large"]
        for i, category in enumerate(categories):
            var = tk.BooleanVar(value=category in cfg.size_categories)
            checkbox_vars[category] = var

            cb = tk.Checkbutton(
                checkbox_container,
                text=category,
                variable=var, bg="#2b2b2b", fg="white", selectcolor="#1e1e1e",
                font=("Arial", 9),
                command=lambda cat=category, v=var: self._toggle_size_category(
                    channel_idx, cat, v, checkbox_vars
                )
            )
            cb.grid(row=i+1, column=0, sticky="w", padx=5, pady=2)

        # Store references for updates
        if not hasattr(self, '_size_checkbox_refs'):
            self._size_checkbox_refs = {}
        self._size_checkbox_refs[channel_idx] = checkbox_vars

    def _create_probability_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Probability:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        self._create_param_slider(frame, 0, 100, cfg.probability, 1,
                                 callback=lambda v: setattr(cfg, 'probability', int(float(v))),
                                 indicator="(Low=Sparse)")

    def _create_modwheel_control(self, parent, channel_idx, cfg):
        """Create modwheel depth control with enable/disable."""
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)

        # Enable checkbox with descriptive label
        enabled_var = tk.BooleanVar(value=cfg.modwheel_enabled)
        cb = tk.Checkbutton(
            frame, text="Modwheel", variable=enabled_var,
            bg="#2b2b2b", fg="white", selectcolor="#1e1e1e",
            command=lambda: setattr(cfg, 'modwheel_enabled', enabled_var.get())
        )
        cb.pack(side=tk.LEFT, padx=5)

        # Depth slider (0-127)
        tk.Label(frame, text="Depth:", bg="#2b2b2b", fg="white", width=7).pack(side=tk.LEFT)
        self._create_param_slider(frame, 0, 127, cfg.modwheel_depth, 1,
                                  callback=lambda v: setattr(cfg, 'modwheel_depth', int(float(v))))

    def _create_pitchbend_control(self, parent, channel_idx, cfg):
        """Create pitchbend depth control with enable/disable."""
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)

        # Enable checkbox with descriptive label
        enabled_var = tk.BooleanVar(value=cfg.pitchbend_enabled)
        cb = tk.Checkbutton(
            frame, text="Pitchbend", variable=enabled_var,
            bg="#2b2b2b", fg="white", selectcolor="#1e1e1e",
            command=lambda: setattr(cfg, 'pitchbend_enabled', enabled_var.get())
        )
        cb.pack(side=tk.LEFT, padx=5)

        # Depth slider (0.0-1.0, display as percentage or semitones)
        tk.Label(frame, text="Depth:", bg="#2b2b2b", fg="white", width=7).pack(side=tk.LEFT)

        # Scale 0-100 slider to 0.0-1.0 internally
        depth_percentage = int(cfg.pitchbend_depth * 100)
        self._create_param_slider(frame, 0, 100, depth_percentage, 1,
                                  callback=lambda v: setattr(cfg, 'pitchbend_depth', float(v) / 100.0))

        # Label showing approximate semitones (assuming ±2 semitones at depth=0.25)
        semitones = cfg.pitchbend_depth * 8  # Rough estimate: 0.25 depth ≈ ±2 semitones
        tk.Label(frame, text=f"(~±{semitones:.1f}st)",
                 bg="#2b2b2b", fg="#888888", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)

    def _create_cc_smoothing_control(self, parent, channel_idx, cfg):
        """Create CC smoothing (laziness) control."""
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Laziness:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)

        # Scale 0-100 slider to 0.0-1.0 internally
        smoothing_percentage = int(cfg.cc_smoothing * 100)
        self._create_param_slider(frame, 0, 100, smoothing_percentage, 1,
                                 callback=lambda v: setattr(cfg, 'cc_smoothing', float(v) / 100.0),
                                 indicator="% (0=instant)")

    def _create_note_min_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Min Note:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        current_note = note_to_name(cfg.min_note)
        note_part = current_note[:-1]
        octave_part = current_note[-1]
        min_note_var = tk.StringVar(value=note_part)
        min_octave_var = tk.StringVar(value=octave_part)
        note_cb = ttk.Combobox(frame, textvariable=min_note_var, values=NOTES, state="readonly", width=5)
        note_cb.pack(side=tk.LEFT, padx=2)
        octave_cb = ttk.Combobox(frame, textvariable=min_octave_var, values=[str(i) for i in range(0, 9)], state="readonly", width=3)
        octave_cb.pack(side=tk.LEFT, padx=2)
        def update_min_note(*args):
            note_name = f"{min_note_var.get()}{min_octave_var.get()}"
            self.update_channel_note_range(channel_idx, "min", note_name)
        min_note_var.trace("w", update_min_note)
        min_octave_var.trace("w", update_min_note)

    def _create_note_max_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Max Note:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        current_note = note_to_name(cfg.max_note)
        note_part = current_note[:-1]
        octave_part = current_note[-1]
        max_note_var = tk.StringVar(value=note_part)
        max_octave_var = tk.StringVar(value=octave_part)
        note_cb = ttk.Combobox(frame, textvariable=max_note_var, values=NOTES, state="readonly", width=5)
        note_cb.pack(side=tk.LEFT, padx=2)
        octave_cb = ttk.Combobox(frame, textvariable=max_octave_var, values=[str(i) for i in range(0, 9)], state="readonly", width=3)
        octave_cb.pack(side=tk.LEFT, padx=2)
        def update_max_note(*args):
            note_name = f"{max_note_var.get()}{max_octave_var.get()}"
            self.update_channel_note_range(channel_idx, "max", note_name)
        max_note_var.trace("w", update_max_note)
        max_octave_var.trace("w", update_max_note)

    def _create_velocity_control(self, parent, channel_idx, cfg):
        frame1 = tk.Frame(parent, bg="#2b2b2b")
        frame1.pack(fill=tk.X, pady=2)
        tk.Label(frame1, text="Min Velocity:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        self._create_param_slider(frame1, 1, 127, cfg.min_velocity, 1, callback=lambda v: setattr(cfg, 'min_velocity', int(float(v))), indicator="(High=Loud)")
        
        frame2 = tk.Frame(parent, bg="#2b2b2b")
        frame2.pack(fill=tk.X, pady=2)
        tk.Label(frame2, text="Max Velocity:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        self._create_param_slider(frame2, 1, 127, cfg.max_velocity, 1, callback=lambda v: setattr(cfg, 'max_velocity', int(float(v))), indicator="(High=Loud)")

    def _create_min_duration_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Min Duration:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        self._create_param_slider(frame, 0, 5000, cfg.min_note_duration, 10,
                                 callback=lambda v: setattr(cfg, 'min_note_duration', int(float(v))),
                                 indicator="ms (0=off)")

    def _create_duration_deviation_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Deviation:", bg="#2b2b2b", fg="white", width=15, anchor="w").pack(side=tk.LEFT)
        self._create_param_slider(frame, 0, 1000, cfg.duration_deviation, 10,
                                 callback=lambda v: setattr(cfg, 'duration_deviation', int(float(v))),
                                 indicator="ms (±)")

    def _create_color_filter_control(self, parent, channel_idx, cfg):
        frame = tk.Frame(parent, bg="#2b2b2b")
        frame.pack(fill=tk.X, pady=2)
        color_enabled = tk.BooleanVar(value=cfg.color_filter_enabled)
        tk.Checkbutton(frame, text="Enable", variable=color_enabled, bg="#2b2b2b", fg="white",
                      selectcolor="#1e1e1e", font=("Arial", 9),
                      command=lambda: setattr(cfg, 'color_filter_enabled', color_enabled.get())).pack(side=tk.LEFT, padx=5)
        eyedropper_btn = tk.Button(frame, text="🎨 Pick Color", font=("Arial", 9, "bold"),
                                   bg="#FF9800", fg="white",
                                   command=lambda: self.activate_eyedropper(channel_idx),
                                   relief=tk.FLAT, cursor="hand2", activebackground="#F57C00")
        eyedropper_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        color_display = tk.Label(frame, text="   ", bg=cfg.display_color, width=3,
                                relief=tk.RAISED, borderwidth=2)
        color_display.pack(side=tk.LEFT, padx=5)

    def update_channel_name(self, channel_idx, name):
        self.channel_configs[channel_idx].name = name

    def apply_size_preset_inline(self, channel_idx, preset_name):
        """DEPRECATED: Kept for backward compatibility.
        New system uses _toggle_size_category() with checkboxes.
        """
        if preset_name in SIZE_PRESETS:
            min_a, max_a = SIZE_PRESETS[preset_name]
            self.channel_configs[channel_idx].min_area = min_a
            self.channel_configs[channel_idx].max_area = max_a
            self.channel_configs[channel_idx].size_filter = preset_name

            # Update new system too
            cfg = self.channel_configs[channel_idx]
            cfg.size_categories = {preset_name}

    def _toggle_size_category(self, channel_idx, category, var, all_checkbox_vars):
        """Handle checkbox toggle with special 'All' logic."""
        cfg = self.channel_configs[channel_idx]
        is_checked = var.get()

        if category == "All":
            if is_checked:
                # "All" checked: select everything
                cfg.size_categories = {"All", "Tiny", "Small", "Medium", "Large"}
                for cat_name, cat_var in all_checkbox_vars.items():
                    cat_var.set(True)
            else:
                # "All" unchecked: deselect everything
                cfg.size_categories.clear()
                for cat_name, cat_var in all_checkbox_vars.items():
                    cat_var.set(False)
        else:
            # Individual category toggled
            if is_checked:
                cfg.size_categories.add(category)
                # Auto-check "All" if all individuals are checked
                if {"Tiny", "Small", "Medium", "Large"}.issubset(cfg.size_categories):
                    cfg.size_categories.add("All")
                    all_checkbox_vars["All"].set(True)
            else:
                cfg.size_categories.discard(category)
                # Uncheck "All" if any individual is unchecked
                cfg.size_categories.discard("All")
                all_checkbox_vars["All"].set(False)

        # Warning for empty category
        if not cfg.size_categories and not is_checked:
            self.log(f"WARNING: {cfg.name} has no categories - will not match any blobs!")
        else:
            self.log(f"{cfg.name}: Categories = {', '.join(sorted(cfg.size_categories)) or 'None'}")

    def activate_eyedropper(self, channel_idx):
        self.eyedropper_active = True
        self.eyedropper_target_channel = channel_idx
        self.canvas.config(cursor="crosshair")
        self.canvas.bind("<Button-1>", self.eyedropper_pick_color)
        self.log(f"Eyedropper active for {self.channel_configs[channel_idx].name} - Click on video to pick color")

    def eyedropper_pick_color(self, event):
        if not self.eyedropper_active or not hasattr(self, 'current_frame_hsv'):
            return
        x, y = event.x, event.y
        if 0 <= x < self.current_frame_hsv.shape[1] and 0 <= y < self.current_frame_hsv.shape[0]:
            h, s, v = self.current_frame_hsv[y, x]
            cfg = self.channel_configs[self.eyedropper_target_channel]
            cfg.color_filter_enabled = True
            cfg.color_hue_min = max(0, h - 10)
            cfg.color_hue_max = min(180, h + 10)
            cfg.color_sat_min = max(0, s - 50)
            cfg.color_sat_max = min(255, s + 50)
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(h/180.0, s/255.0, v/255.0)
            color_hex = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            cfg.display_color = color_hex
            for ui_data in self.active_channel_uis:
                if ui_data["index"] == self.eyedropper_target_channel:
                    if "name_entry" in ui_data["widgets"]:
                        ui_data["widgets"]["name_entry"].config(bg=color_hex)
                    break
            self.log(f"Color picked for {cfg.name}")
        self.eyedropper_active = False
        self.canvas.config(cursor="")
        self.canvas.unbind("<Button-1>")

    def update_channel_midi(self, channel_idx, midi_channel):
        self.channel_configs[channel_idx].midi_channel = midi_channel
        # Update the MIDI channel label in the header (display 1-16)
        for ui_data in self.active_channel_uis:
            if ui_data["index"] == channel_idx:
                ui_data["widgets"]["midi_ch_label"].config(text=f"Ch{midi_channel + 1}")

    def update_channel_note_range(self, channel_idx, min_or_max, note_name):
        note_num = name_to_note(note_name)
        if min_or_max == "min":
            self.channel_configs[channel_idx].min_note = note_num
        else:
            self.channel_configs[channel_idx].max_note = note_num

    def remove_channel_ui(self, channel_idx):
        self.channel_configs[channel_idx].enabled = False
        for i, ui_data in enumerate(self.active_channel_uis):
            if ui_data["index"] == channel_idx:
                ui_data["widgets"]["container"].destroy()
                self.active_channel_uis.pop(i)
                break
        self.log(f"Removed {self.channel_configs[channel_idx].name}")

    # ========================================================================
    # CORE LOGIC
    # ========================================================================

    def log(self, msg: str):
        if self.log_box is None:
            # Log not created yet, skip silently
            return
        self.log_box.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see(tk.END)
        if int(self.log_box.index('end-1c').split('.')[0]) > 500:
            self.log_box.delete(1.0, 100.0)

    def set_src(self, t: str):
        self.cap.release()
        if t == "file":
            p = filedialog.askopenfilename()
            if p:
                self.cap = cv2.VideoCapture(p)
                self.log(f"Video: {os.path.basename(p)}")
            else:
                self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(0)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.bg_history_var.get(), 
                                                       varThreshold=self.bg_threshold_var.get(), 
                                                       detectShadows=False)

    def set_src_button(self, t: str):
        self.set_src(t)
        if t == "cam":
            self.btn_webcam.config(bg="#2196F3", font=("Arial", 9, "bold"))
            self.btn_file.config(bg="#2b2b2b", font=("Arial", 9))
            self.video_source_var.set("Webcam")
        else:
            self.btn_file.config(bg="#2196F3", font=("Arial", 9, "bold"))
            self.btn_webcam.config(bg="#2b2b2b", font=("Arial", 9))
            self.video_source_var.set("File")

    def toggle_midi(self):
        self.midi_active = not self.midi_active
        if self.midi_active:
            self.btn_run.config(bg="#f44336", text="⏹ STOP MIDI", activebackground="#e53935")
            self.status_midi.config(text="MIDI: Running ●", fg="#4CAF50")
            self.log("MIDI Active - Motion detection running")
        else:
            self.btn_run.config(bg="#4CAF50", text="▶ START MIDI", activebackground="#45a049")
            self.status_midi.config(text="MIDI: Ready ○", fg="#888")
            self.midi.panic()
            self.log("MIDI Stopped")

    def update_scale_pool(self):
        r_idx = NOTES.index(self.root_var.get())
        intervals = SCALES[self.scale_var.get()]
        full_pool = []
        for octave in [2, 3, 4, 5]:
            base = 12 * (octave + 1) + r_idx
            for iv in intervals:
                full_pool.append(base + iv)
        mask_indices = CHORD_MASKS[self.chord_var.get()]
        constrained_pool = []
        scale_len = len(intervals)
        for i, note in enumerate(full_pool):
            degree = i % scale_len
            if degree in mask_indices:
                constrained_pool.append(note)
        self.active_scale_notes = sorted(list(set(constrained_pool)))
        self.log(f"Scale Pool: {len(self.active_scale_notes)} notes")

    def update_midi_monitor(self):
        self.window.after(200, self.update_midi_monitor)

    def capture_preset(self) -> dict:
        return {
            "name": "preset",
            "parameters": {
                "midi_output_port": self.port_var.get(),
                "root_note": self.root_var.get(),
                "scale": self.scale_var.get(),
                "chord_mask": self.chord_var.get(),
                "min_blob_size": self.var_size.get(),
                "trigger_sensitivity": self.var_trigger.get(),
                "bg_threshold": self.bg_threshold_var.get(),
                "bg_history": self.bg_history_var.get(),
                "velocity_mode": self.vel_mode_var.get(),
                "min_velocity": self.var_vel_min.get(),
                "max_velocity": self.var_vel_max.get(),
                "velocity_curve": self.vel_curve_var.get(),
            },
            "channels": [cfg.to_dict() for cfg in self.channel_configs]
        }

    def save_preset(self):
        filepath = filedialog.asksaveasfilename(
            initialdir="./presets",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
            title="Save Preset"
        )
        if not filepath:
            return

        filename = os.path.basename(filepath)
        preset_data = self.capture_preset()
        preset_data["name"] = filename.replace('.json', '')

        if self.preset_manager.save_preset(filename, preset_data):
            self.log(f"Saved preset: {filename}")
        else:
            self.log("Failed to save preset")

    def load_preset(self):
        name = filedialog.askopenfilename(initialdir="./presets", filetypes=[("JSON", "*.json")])
        if name:
            preset_data = self.preset_manager.load_preset(os.path.basename(name))
            if preset_data:
                self.apply_preset(preset_data)

    def apply_preset(self, preset_data: dict):
        if not preset_data or "parameters" not in preset_data:
            return
        params = preset_data["parameters"]

        # Load MIDI output port if available
        midi_port = params.get("midi_output_port")
        if midi_port:
            available_ports = self.midi.get_ports()
            if midi_port in available_ports:
                self.port_var.set(midi_port)
                self.midi.open_port(midi_port)

        self.root_var.set(params.get("root_note", "C"))
        self.scale_var.set(params.get("scale", "Minor Pentatonic"))
        self.chord_var.set(params.get("chord_mask", "Full Scale"))
        self.var_size.set(params.get("min_blob_size", 300))
        self.var_trigger.set(params.get("trigger_sensitivity", 2.0))

        self.bg_threshold_var.set(params.get("bg_threshold", 25))
        self.bg_history_var.set(params.get("bg_history", 500))
        
        self.vel_mode_var.set(params.get("velocity_mode", "Fixed"))
        self.var_vel_min.set(params.get("min_velocity", 40))
        self.var_vel_max.set(params.get("max_velocity", 100))
        self.vel_curve_var.set(params.get("velocity_curve", "Linear"))
        
        for ui_data in list(self.active_channel_uis):
             ui_data["widgets"]["container"].destroy()
        self.active_channel_uis = []

        if "channels" in preset_data:
            for i, ch_data in enumerate(preset_data["channels"]):
                if i < len(self.channel_configs):
                    cfg = ChannelConfig.from_dict(ch_data)
                    self.channel_configs[i] = cfg
                    if cfg.enabled:
                        self.add_channel_ui(target_idx=i)
                        
        self.update_scale_pool()

    def _try_load_default_preset(self):
        if "default.json" in self.preset_manager.list_presets():
            preset_data = self.preset_manager.load_preset("default.json")
            if preset_data:
                self.apply_preset(preset_data)

    def detect_collisions(self):
        blobs = list(self.blobs.values())
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                b1, b2 = blobs[i], blobs[j]
                if b1.collision_cooldown > 0 or b2.collision_cooldown > 0:
                    continue
                r1 = math.sqrt(b1.smooth_area) / 2
                r2 = math.sqrt(b2.smooth_area) / 2
                dist = math.hypot(b1.smooth_x - b2.smooth_x, b1.smooth_y - b2.smooth_y)
                if dist < (r1 + r2):
                    collision_note = 84 + random.randint(0, 12)
                    self.midi.note_on(ACCENT_CHANNEL, collision_note, 120)
                    self.midi.note_off(ACCENT_CHANNEL, collision_note)
                    b1.collision_cooldown = 1.0
                    b2.collision_cooldown = 1.0

    def _generate_channel_notes(self, cfg: ChannelConfig) -> List[int]:
        """Generate note pool for a specific channel (respects overrides).

        Args:
            cfg: ChannelConfig to generate notes for

        Returns:
            List of MIDI note numbers available for this channel
        """
        if cfg.override_root or cfg.override_scale:
            root = cfg.override_root if cfg.override_root else self.root_var.get()
            scale = cfg.override_scale if cfg.override_scale else self.scale_var.get()
            r_idx = NOTES.index(root)
            intervals = SCALES[scale]
            channel_notes = []
            for octave in [2, 3, 4, 5]:
                base = 12 * (octave + 1) + r_idx
                for iv in intervals:
                    channel_notes.append(base + iv)
            channel_notes = [n for n in channel_notes if cfg.min_note <= n <= cfg.max_note]
        else:
            channel_notes = [n for n in self.active_scale_notes if cfg.min_note <= n <= cfg.max_note]

        if not channel_notes:
            channel_notes = [cfg.min_note]

        return channel_notes

    def process_logic(self, dt: float, frame_hsv=None):
        if not self.active_scale_notes:
            return
        w, h = 640, 480
        move_thresh = self.var_trigger.get()

        for b in self.blobs.values():
            b.lfo_phase += dt * b.lfo_rate
            lfo_val = math.sin(b.lfo_phase)
            is_moving = b.smooth_speed > move_thresh

            # Update color fade timer (caps at 20ms)
            if b.color_fade_timer < 0.02:  # 20ms in seconds
                b.color_fade_timer += dt

            if b.state == "OFF":
                if is_moving:
                    b.state = "ATTACK"
                    b.state_timer = 0.0
                    b.color_fade_timer = 0.0  # Reset fade timer for 20ms transition
            elif b.state == "ATTACK":
                # Lock channel assignments and trigger notes immediately
                if not b.channel_configs:
                    b.update_size_categories()

                    # Find ALL matching channels
                    matching_configs = []
                    for cfg in self.channel_configs:
                        if cfg.enabled and cfg.matches_blob(b, frame_hsv):
                            matching_configs.append(cfg)

                    # Lock for blob's lifetime
                    b.channel_configs = matching_configs

                    # Trigger notes immediately on all locked channels (independent probability)
                    for cfg in b.channel_configs:
                        # Per-channel probability gate
                        if random.randint(0, 100) < cfg.probability:
                            channel_notes = self._generate_channel_notes(cfg)

                            # Check if arpeggiator is active for this channel
                            if cfg.arp_mode != "OFF" and cfg.config_id in b.arp_states:
                                # Get next note from arpeggiator sequence
                                arp_state = b.arp_states[cfg.config_id]
                                target = get_next_arp_note(arp_state, cfg.arp_mode, arp_state['note_pool'])
                            else:
                                # Normal behavior: calculate note from blob position
                                note_idx = int((b.smooth_x / w) * len(channel_notes))
                                note_idx = max(0, min(len(channel_notes)-1, note_idx))
                                target = channel_notes[note_idx]

                            velocity = calculate_velocity(b, self.vel_mode_var.get(),
                                                         cfg.min_velocity, cfg.max_velocity,
                                                         self.vel_curve_var.get(),
                                                         blend=0.5, size_range=(self.var_size.get(), 5000),
                                                         speed_range=(0, 50))

                            self.midi.note_on(cfg.midi_channel, target, velocity)
                            b.playing_notes[cfg.config_id] = target
                            b.note_start_times[cfg.config_id] = time.time()  # Track start time for duration enforcement

                            # Reset smoothed CC values for this channel to prevent flickering
                            # when retriggering with different modulation values
                            if cfg.config_id in b.smoothed_cc_values:
                                del b.smoothed_cc_values[cfg.config_id]

                            # Initialize arpeggiator state if enabled
                            if cfg.arp_mode != "OFF":
                                # Initialize arp state
                                arp_state = {
                                    'index': 0,
                                    'direction': 1,  # Start ascending
                                    'note_pool': channel_notes if channel_notes else [target]
                                }

                                # For DOWN mode, start at the end
                                if cfg.arp_mode == "DOWN":
                                    arp_state['index'] = len(arp_state['note_pool']) - 1

                                b.arp_states[cfg.config_id] = arp_state

                    b.state = "ON"

                if not is_moving:
                    b.state = "OFF"
                    b.channel_configs = []  # Release if movement stops
            elif b.state == "ON":
                if not is_moving:
                    b.state = "RELEASE"
                    b.state_timer = 0.0
                else:
                    # Only process CC if at least one note is playing
                    if b.playing_notes:
                        # Calculate shared values once
                        pan = int((b.smooth_x / w) * 127)
                        filt = int((1 - (b.smooth_y / h)) * 127)
                        base_vol = 40 + min(60, b.smooth_speed * 5)
                        breath = lfo_val * 10
                        final_vol = int(max(0, min(127, base_vol + breath)))

                        # Send to all channels with active notes
                        for cfg in b.channel_configs:
                            if cfg.config_id in b.playing_notes:
                                # Initialize smoothed CC storage for this config if needed
                                if cfg.config_id not in b.smoothed_cc_values:
                                    b.smoothed_cc_values[cfg.config_id] = {}

                                # Smoothing factor (0=instant, 1=very lazy)
                                # Calculate time-based smoothing coefficient
                                # At 0% smoothing: instant (tau very small)
                                # At 100% smoothing: very slow (tau = 1 second)
                                smooth_factor = cfg.cc_smoothing
                                tau = 0.001 + (smooth_factor * 0.999)  # Time constant from 1ms to 1000ms
                                alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0  # Time-based smoothing coefficient

                                # Helper to smooth a CC value with time-based exponential filter
                                def smooth_cc(cc_num, target_value):
                                    if cc_num not in b.smoothed_cc_values[cfg.config_id]:
                                        b.smoothed_cc_values[cfg.config_id][cc_num] = float(target_value)
                                    else:
                                        current = b.smoothed_cc_values[cfg.config_id][cc_num]
                                        # Time-based exponential smoothing
                                        b.smoothed_cc_values[cfg.config_id][cc_num] = current + alpha * (target_value - current)
                                    return int(b.smoothed_cc_values[cfg.config_id][cc_num])

                                # Modulation CC
                                if cfg.modwheel_enabled and cfg.modwheel_depth > 0:
                                    modwheel_val = calculate_modulation(b, lfo_val, "modwheel", cfg.modwheel_depth)
                                    smoothed_modwheel = smooth_cc(1, modwheel_val)
                                    self.midi.send_cc(cfg.midi_channel, 1, smoothed_modwheel)

                                if cfg.pitchbend_enabled and cfg.pitchbend_depth > 0:
                                    pitchbend_val = calculate_modulation(b, lfo_val, "pitchbend", cfg.pitchbend_depth)
                                    # Pitchbend smoothing using same alpha
                                    if 'pitchbend' not in b.smoothed_cc_values[cfg.config_id]:
                                        b.smoothed_cc_values[cfg.config_id]['pitchbend'] = float(pitchbend_val)
                                    else:
                                        current_pb = b.smoothed_cc_values[cfg.config_id]['pitchbend']
                                        b.smoothed_cc_values[cfg.config_id]['pitchbend'] = current_pb + alpha * (pitchbend_val - current_pb)
                                    smoothed_pitchbend = int(b.smoothed_cc_values[cfg.config_id]['pitchbend'])
                                    self.midi.send_pitchbend(cfg.midi_channel, smoothed_pitchbend)

                                # Positional CC with smoothing
                                smoothed_pan = smooth_cc(10, pan)
                                smoothed_filt = smooth_cc(74, filt)
                                smoothed_vol = smooth_cc(11, final_vol)

                                self.midi.send_cc(cfg.midi_channel, 10, smoothed_pan)
                                self.midi.send_cc(cfg.midi_channel, 74, smoothed_filt)
                                self.midi.send_cc(cfg.midi_channel, 11, smoothed_vol)
            elif b.state == "RELEASE":
                if is_moving:
                    b.state = "ON"
                else:
                    # Release notes immediately once minimum duration is met
                    current_time = time.time()
                    notes_to_release = []

                    for cfg in b.channel_configs:
                        if cfg.config_id in b.playing_notes:
                            # Check if minimum duration has elapsed
                            can_release = True
                            if cfg.min_note_duration > 0 and cfg.config_id in b.note_start_times:
                                # Calculate required duration with random deviation
                                deviation = random.uniform(-cfg.duration_deviation, cfg.duration_deviation)
                                required_duration = (cfg.min_note_duration + deviation) / 1000.0  # Convert ms to seconds

                                elapsed = current_time - b.note_start_times[cfg.config_id]
                                if elapsed < required_duration:
                                    can_release = False

                            if can_release:
                                note = b.playing_notes[cfg.config_id]
                                self.midi.note_off(cfg.midi_channel, note)

                                if cfg.pitchbend_enabled:
                                    self.midi.send_pitchbend(cfg.midi_channel, 0)

                                notes_to_release.append(cfg.config_id)

                    # Remove released notes from tracking
                    for config_id in notes_to_release:
                        del b.playing_notes[config_id]
                        if config_id in b.note_start_times:
                            del b.note_start_times[config_id]

                    # Only transition to OFF if all notes have been released
                    if not b.playing_notes:
                        b.channel_configs = []
                        b.state = "OFF"

    def update_loop(self):
        curr_time = time.time()
        dt = curr_time - self.last_time
        self.last_time = curr_time

        # Update keyboard fade states for smooth transitions
        self.update_key_fades(dt, fade_speed=10.0)

        self.frame_count += 1
        if curr_time - self.fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_time = curr_time
            self.status_fps.config(text=f"{self.fps} FPS")

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)

            self.fgbg.setVarThreshold(float(self.bg_threshold_var.get()))
            self.fgbg.setHistory(int(self.bg_history_var.get()))

            brightness = self.var_brightness.get()
            contrast = self.var_contrast.get() / 100.0
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.current_frame_hsv = frame_hsv

            blur = cv2.GaussianBlur(frame, (21, 21), 0)
            mask = self.fgbg.apply(blur)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets = []
            min_a = self.var_size.get()

            for c in contours:
                area = cv2.contourArea(c)
                if area > min_a:
                    M = cv2.moments(c)
                    if M["m00"]:
                        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    else:
                        cx, cy = 0, 0
                    dets.append((cx, cy, area))
                    hull = cv2.convexHull(c)
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

            self.track_blobs(dets)
            if self.midi_active:
                self.process_logic(dt, frame_hsv)
                self.detect_collisions()

            for b in self.blobs.values():
                if len(b.trail) > 1:
                    pts = np.array(b.trail, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (0, 255, 255), 1)

                # State-based color for blob border with 20ms fade to red
                # Use customizable colors if available, otherwise fallback to defaults
                off_color = getattr(self, 'blob_color_off', BLOB_COLOR_OFF)
                on_color = getattr(self, 'blob_color_on', BLOB_COLOR_ON)

                if b.state == "OFF":
                    blob_color = off_color
                elif b.state in ("ATTACK", "ON", "RELEASE"):
                    # Fade from OFF color to ON color over 20ms
                    fade_progress = min(1.0, b.color_fade_timer / 0.02)  # 0.02 = 20ms
                    blob_color = interpolate_color(off_color, on_color, fade_progress)
                else:
                    blob_color = (128, 128, 128)  # Gray fallback

                # Draw blob border with state-based color (fine 1px border)
                cv2.circle(frame, (int(b.smooth_x), int(b.smooth_y)),
                           int(math.sqrt(b.smooth_area)/2) + 1, blob_color, 1)

                # Minimalistic channel indicator - only show when notes are playing
                if b.channel_configs and self.notes_toggle.get() and b.playing_notes:
                    # Show how many channels are actively playing notes
                    active_count = len(b.playing_notes)

                    if active_count == 1:
                        # Single note playing: show which channel
                        config_id = list(b.playing_notes.keys())[0]
                        for cfg in b.channel_configs:
                            if cfg.config_id == config_id:
                                ch_text = f"Ch{cfg.midi_channel+1}"
                                break
                    else:
                        # Multiple notes: show count of active channels
                        ch_text = f"{active_count} Ch"

                    # Single line of text, small and clean
                    cv2.putText(frame, ch_text,
                                (int(b.smooth_x)-15, int(b.smooth_y)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb_frame)

            if self.canvas_width > 10 and self.canvas_height > 10:
                scale_w = self.canvas_width / pil_img.width
                scale_h = self.canvas_height / pil_img.height
                scale = min(scale_w, scale_h)

                new_width = int(pil_img.width * scale)
                new_height = int(pil_img.height * scale)

                pil_img = pil_img.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
                x_offset = (self.canvas_width - new_width) // 2
                y_offset = (self.canvas_height - new_height) // 2

                img = PIL.ImageTk.PhotoImage(image=pil_img)
                self.canvas.delete("all")
                self.canvas.create_image(x_offset, y_offset, image=img, anchor=tk.NW)
                self.canvas.image = img
            else:
                img = PIL.ImageTk.PhotoImage(image=pil_img)
                self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
                self.canvas.image = img

        self.window.after(30, self.update_loop)

    def track_blobs(self, dets):
        used = set()
        active = []
        for bid, b in self.blobs.items():
            best_d = 150
            best_i = -1
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = math.hypot(b.x - d[0], b.y - d[1])
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i != -1:
                b.update(*dets[best_i], 0.1)
                used.add(best_i)
                active.append(bid)

        for bid in list(self.blobs.keys()):
            if bid not in active:
                blob = self.blobs[bid]

                # Send note_off for all active notes before deletion
                if blob.playing_notes and self.midi.outport:
                    for cfg in blob.channel_configs:
                        if cfg.config_id in blob.playing_notes:
                            note = blob.playing_notes[cfg.config_id]
                            self.midi.note_off(cfg.midi_channel, note)

                del self.blobs[bid]

        for i, d in enumerate(dets):
            if i not in used and len(self.blobs) < 15:
                self.blobs[self.next_id] = BlobEntity(self.next_id, *d)
                self.next_id += 1

    def __del__(self):
        self.midi.panic()
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "MAGMA v7.7 - Professional MIDI Generator")
    root.mainloop()