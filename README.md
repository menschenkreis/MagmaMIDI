# MAGMA - Motion-Activated Generative MIDI Application

MAGMA is a professional blob detection to MIDI conversion system that transforms motion detected from a webcam or video file into musical output. Using computer vision to track moving objects, MAGMA maps their position, size, and speed to MIDI notes, velocities, and modulation parameters in real-time.

## Overview

MAGMA bridges the gap between physical motion and digital music creation. Whether you are a performer looking for an interactive instrument, an installation artist seeking responsive audio, or simply experimenting with motion-to-sound mapping, MAGMA provides a flexible and powerful platform. The application processes video input through OpenCV's background subtraction algorithms to identify moving blobs, then maps each blob's characteristics to musical parameters across multiple MIDI channels.

The core philosophy behind MAGMA is accessibility without sacrificing capability. The interface presents complex signal processing and musical mapping in an intuitive three-panel layout, while advanced users can fine-tune every parameter to achieve precise control over the generated output. From simple single-channel note triggering to complex multi-timbral arrangements with independent arpeggiators and modulation chains, MAGMA scales with your creative ambitions.

## Features

MAGMA encompasses a comprehensive suite of motion tracking and musical mapping tools designed for both immediate playability and deep customization.

### Motion Detection and Tracking

The application employs OpenCV's BackgroundSubtractorMOG2 algorithm to distinguish moving objects from static backgrounds. This approach efficiently isolates blobs of interest while filtering out camera noise and minor environmental changes. The detection sensitivity is fully adjustable through threshold and history parameters, allowing you to fine-tune the system's response to your specific lighting conditions and desired detection granularity.

Blob tracking maintains persistent identities across frames using distance-based matching, enabling the system to follow individual objects as they move through the scene. Each tracked blob accumulates velocity data, size measurements, and positional history, which feed into the musical mapping engines. The tracking system supports up to 15 simultaneous blobs, with older or lost blobs cleanly removed from the active pool.

### Multi-Channel MIDI Architecture

MAGMA operates on a 16-channel MIDI architecture where each channel maintains independent configuration parameters. This design enables sophisticated musical arrangements where different blob sizes trigger different instruments, color-filtered objects play specific timbres, or complex layered textures emerge from the interaction of multiple tracking targets.

Each channel can be individually enabled or disabled, assigned a unique MIDI output channel, named for organizational clarity, and configured with specific note ranges, velocity profiles, and probability gates. The channel system supports hierarchical assignment where blobs can lock onto multiple channels simultaneously upon detection, creating rich harmonic content from simple motion events.

### Musical Scale and Mapping System

The application implements a sophisticated scale engine built on Western music theory. Eight distinct scale formulas are available: Major, Minor, Minor Pentatonic, Major Pentatonic, Dorian, Lydian, Mixolydian, and Phrygian. Each scale produces a pool of available MIDI notes that are mapped to horizontal blob position, creating an intuitive pitch array across the video frame.

Chord mask options further refine the note selection, allowing you to restrict output to specific scale degrees. Options include full scale playback, triads (root-third-fifth), seventh chords, and open fifths. This flexibility accommodates everything atonal experimental textures to diatonic melodic playing.

### Dynamic Velocity Processing

Velocity determination supports multiple algorithms to suit different performance contexts. Fixed velocity mode produces consistent loudness regardless of blob characteristics. Size-based velocity calculates loudness proportional to blob area, so larger movements generate louder notes. Speed-based velocity links loudness to movement velocity, rewarding energetic motion with increased volume. Combined mode blends both parameters with configurable weighting.

Each velocity mode includes selectable response curves: linear provides straightforward mapping, exponential emphasizes extreme values for dramatic effect, and S-curve creates smooth transitions with a centered neutral point. The velocity processor respects minimum and maximum thresholds to ensure output stays within practical MIDI ranges while maintaining musical dynamics.

### Advanced Modulation System

MAGMA's modulation engine transforms blob motion into expressive control changes. Modwheel (MIDI CC1) output responds to blob speed combined with an internal LFO, creating oscillating timbral changes that add movement and life to sustained notes. The modwheel depth parameter controls the maximum modulation intensity, while the smoothing factor prevents jarring parameter jumps.

Pitchbend modulation applies similar principles to pitch, with blob speed influencing subtle pitch drift and the LFO creating organic vibrato-like effects. The pitchbend depth scales the effect from subtle inflections to pronounced pitch swings spanning several semitones. CC smoothing parameters apply exponential filtering to all continuous controller outputs, eliminating digital artifacts while maintaining responsive real-time control.

### Integrated Arpeggiator

Each MIDI channel includes a built-in arpeggiator that transforms static note triggering into rhythmic patterns. When enabled, the arpeggiator cycles through available notes in one of four patterns: UP steps through the note pool sequentially, DOWN traverses in reverse, UP-DOWN creates ping-pong patterns that reverse direction at boundaries, and RANDOM selects notes unpredictably for generative textures.

The arpeggiator maintains independent state per blob per channel, meaning each tracked object can run its own arpeggio pattern independently. This capability enables complex polyrhythmic textures where multiple blobs generate interlocking patterns across different timing relationships.

### Visual Piano Interface

A virtual keyboard display at the bottom of the interface provides real-time feedback of active MIDI notes. Keys illuminate when notes sound, with color coding indicating which channel triggered each note. The keyboard supports mouse interaction for manual testing, allowing you to verify sound engine response without motion input.

### Preset Management System

Configuration presets save complete system state including global parameters, channel configurations, and MIDI port selections. Presets store to JSON files in a local presets directory, enabling quick switching between different setups for different performances or experiments. The default preset loads automatically on startup, providing a consistent starting point.

## Requirements

MAGMA depends on several Python packages that provide computer vision, MIDI, and graphical user interface capabilities. The following requirements must be satisfied before running the application:

- **Python 3.8 or higher** - The application uses modern Python features and type hints throughout
- **OpenCV (cv2)** - Provides video capture, image processing, and background subtraction algorithms
- **PIL (Pillow)** - Handles image format conversion and resizing for the interface display
- **mido** - Manages MIDI port communication with external synthesizers and software
- **NumPy** - Supports numerical operations for array processing and mathematical calculations

Additional tkinter support is included with standard Python installations on most platforms.

## Installation

Begin by cloning or downloading the MAGMA repository to your local system. Navigate to the project directory and install the required Python packages using pip or your preferred package manager.

```bash
pip install opencv-python pillow mido numpy
```

For full MIDI functionality, you may need platform-specific MIDI drivers. On Windows, the native Windows MIDI drivers should function automatically. On macOS, install the portmidi library through Homebrew:

```bash
brew install portmidi
pip install portmidi
```

On Linux, ensure ALSA support is installed:

```bash
sudo apt-get install libasound2-dev
pip install pyalsa
```

## Running the Application

Launch MAGMA from the terminal by executing the main script:

```bash
python magma.py
```

The application window will appear showing the three-panel interface with motion detection active. Initially, no MIDI output occurs until you select an output port and enable MIDI processing. The status bar at the bottom indicates system state with MIDI readiness and FPS counters.

## Interface Guide

The MAGMA interface presents motion video, system controls, and channel configurations in a resizable three-column layout.

### Left Sidebar - Global Controls

The left panel contains system-wide parameters that affect all motion detection and musical processing. The header displays the application version followed by global configuration sections organized into collapsible panels.

**Video Source Section**
- Webcam button activates the primary camera for motion input
- File button opens video file selection for pre-recorded input
- Brightness slider adjusts input video exposure
- Contrast slider modifies input video dynamic range
- Threshold slider controls background subtraction sensitivity
- History slider sets the number of frames used for background modeling

**Scale Settings Section**
- Root note selector determines the scale's tonic pitch
- Scale selector chooses the musical scale formula
- Chord mask selector restricts notes to specific scale degrees

**Velocity Settings Section**
- Mode selector chooses velocity calculation method
- Curve selector adjusts velocity response curve
- Minimum and maximum sliders constrain velocity range

### Center Panel - Video Display

The central panel displays the processed video feed with motion detection visualization. Moving blobs appear with color-coded borders indicating their current state: green for idle, transitioning through orange during attack phases, red while actively playing, and yellow during release phases.

The canvas supports color picking when the eyedropper tool is active. Click any location to sample the HSV color values, which can then be applied to channel color filters.

### Right Sidebar - Channel Configuration

The right panel contains MIDI channel configuration panels. Each channel appears as a collapsible section with the channel name, MIDI channel assignment, and enable checkbox in the header. The channel content expands to reveal configuration sections.

**Basic Section**
- MIDI Channel selector assigns the output channel (1-16)
- Size category checkboxes select which blob sizes trigger this channel
- Probability slider determines the percentage chance of triggering

**Note Range Section**
- Minimum and maximum note selectors define the playable pitch range

**Velocity Section**
- Minimum and maximum velocity thresholds constrain note intensity

**Note Duration Section**
- Minimum duration prevents note cutting with brief triggers
- Deviation adds random timing variation to minimum duration

**Color Filter Section**
- Enable checkbox activates HSV color filtering
- Pick Color button opens the eyedropper for color sampling

**Modulation Section**
- Modwheel enable and depth controls CC1 output
- Pitchbend enable and depth controls pitch wheel output
- Laziness slider adjusts CC smoothing response

## Configuration Reference

This section provides detailed documentation of all configurable parameters for reference during setup and performance.

### Global Parameters

The global parameters establish baseline behavior for the motion detection and musical mapping systems.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Background Threshold | 0-255 | 16 | Sensitivity of background subtraction |
| Background History | 1-2000 | 500 | Frames used for background model |
| Brightness | -100 to 100 | 0 | Input video brightness adjustment |
| Contrast | 0-200 | 100 | Input video contrast multiplier |
| Min Blob Size | 0-10000 | 300 | Minimum blob area for detection |
| Trigger Sensitivity | 0.1-10.0 | 2.0 | Speed threshold for note triggering |

### Scale Definitions

MAGMA implements scales through interval patterns relative to the root note. The available scale formulas use standard Western music theory notation:

| Scale Name | Intervals | Character |
|------------|-----------|-----------|
| Major | 0, 2, 4, 5, 7, 9, 11 | Bright, happy |
| Minor | 0, 2, 3, 5, 7, 8, 10 | Dark, sad |
| Minor Pentatonic | 0, 3, 5, 7, 10 | Bluesy, rock |
| Major Pentatonic | 0, 2, 4, 7, 9 | Folk, modal |
| Dorian | 0, 2, 3, 5, 7, 9, 10 | Jazz, funky |
| Lydian | 0, 2, 4, 6, 7, 9, 11 | Dreamy, ethereal |
| Mixolydian | 0, 2, 4, 5, 7, 9, 10 | Rock, blues |
| Phrygian | 0, 1, 3, 5, 7, 8, 10 | Spanish, metal |

### Size Category Definitions

Blob size categories classify detected objects based on pixel area for selective triggering:

| Category | Area Range | Use Case |
|----------|------------|----------|
| Tiny | 50-500 | Fingers, small objects |
| Small | 500-1500 | Hands, moderate motion |
| Medium | 1500-3000 | Full body, large gestures |
| Large | 3000-10000 | Multiple people, expansive motion |

### MIDI Implementation

MAGMA sends standard MIDI messages compatible with any MIDI-capable device or software synthesizer. The following messages are generated:

**Note Messages**
- note_on triggered when motion begins on a matching blob
- note_off triggered when motion ceases and minimum duration elapses
- Notes are clamped to 0-127 range

**Control Change Messages**
- CC1 (Modulation Wheel) - Blob speed + LFO modulated
- CC10 (Pan) - Horizontal position mapped
- CC11 (Expression) - Speed-based volume
- CC74 (Filter Cutoff) - Vertical position mapped

**Pitch Bend**
- Pitchbend values from -8192 to +8191
- Center (0) represents no bend

## Usage Examples

The following examples demonstrate common configurations for different performance scenarios.

### Basic Installation Setup

For a fixed installation where performers enter from a specific area, configure a single channel with a tight color filter matching performer clothing. Set the size category to Medium or Large to ignore small background movements. Use the Minor Pentatonic scale with chord mask set to Seventh for rich harmonic content. Enable pitchbend modulation with moderate depth to add expression to sustained movements.

### Interactive Performance Setup

For real-time performance with hand motion, enable multiple channels targeting different size categories simultaneously. Configure the Tiny category for high-pitched melodic lines, Small for mid-range accompaniment, and Medium for bass notes. Set probability to 100% on all channels for immediate response. Enable arpeggiator on the melody channel with UP-DOWN mode for flowing patterns.

### Generative Installation Setup

For ambient installations, configure multiple channels with overlapping but not identical settings. Use different scales on each channel to create harmonic tension and resolution over time. Set probability to 30-50% for sporadic triggering. Enable CC smoothing at high values (80-100%) for slow, evolving modulation. Disable visual elements to reduce distraction from the sonic output.

## Troubleshooting

If MAGMA does not respond as expected, the following solutions address common issues.

**No MIDI Output**
- Verify a valid MIDI output port is selected
- Confirm the destination device or software is properly connected
- Check that MIDI is enabled with the START MIDI button
- Ensure channel configurations are enabled

**No Motion Detection**
- Check camera permissions and availability
- Reduce the background threshold value
- Increase the minimum blob size to reduce noise
- Verify sufficient contrast between subject and background

**Erratic Note Triggering**
- Increase trigger sensitivity threshold
- Reduce probability values for selective triggering
- Enable color filtering to isolate specific objects
- Add minimum duration to prevent choppy notes

**High CPU Usage**
- Reduce video resolution in the capture settings
- Decrease background history value
- Limit the number of enabled channels
- Close other applications using camera or MIDI resources

## Architecture Notes

MAGMA follows a modular architecture separating concerns into distinct components. The application initializes with a MidiEngine handling all MIDI communication, a PresetManager managing configuration persistence, and the main App class coordinating video processing and musical mapping.

Video frames undergo background subtraction to generate a motion mask, which is then processed through morphological operations to clean noise. Contours are extracted from the mask and converted to blob detections with positional and area data. The tracking system maintains blob identities across frames using distance-based matching.

Musical processing occurs in the process_logic method, which evaluates each blob's state machine transitions (OFF → ATTACK → ON → RELEASE). The state machine determines when notes trigger, sustain, and release based on motion detection and configurable minimum durations. Arpeggiator logic runs for enabled channels, cycling through note pools independently per blob.

All visual rendering occurs in the update_loop method, which composites the video frame with blob visualizations and the virtual piano keyboard. The rendering pipeline maintains 30 FPS target for smooth visual feedback.

## Contributing

Contributions to MAGMA are welcome from developers and musicians alike. Areas of particular interest include additional scale definitions, enhanced visualization options, OSC output support, and performance optimizations.

Before submitting pull requests, ensure code follows the existing style conventions and include documentation for new features. Test thoroughly with different MIDI devices and video sources to verify cross-platform compatibility.

## License

MAGMA is provided under the MIT License, permitting commercial use, modification, and distribution with appropriate attribution. See the LICENSE file for complete terms.

## Acknowledgments

MAGMA builds upon the excellent work of the OpenCV community, the mido MIDI library maintainers, and the broader Python scientific computing ecosystem. Special thanks to contributors who have tested, reported issues, and suggested improvements throughout the project's development.
