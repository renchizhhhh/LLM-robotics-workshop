# Spot FSM Control V2

Een nieuwe, schone implementatie van Spot robot control met JSON-gebaseerde commando's en een Finite State Machine.

## Overzicht

De V2 implementatie bestaat uit vier hoofdbestanden:

1. **`finite_state_machine.py`** - FSM met minimale staten (sit, stand, action)
2. **`spot_control_interface.py`** - Interface voor alle Spot SDK calls
3. **`fsm_node.py`** - ROS node die JSON commando's verwerkt
4. **`natural_language_control.py`** - Vertaler van NL naar JSON acties

## Belangrijkste Verbeteringen

- **Elke robotactie is precies één API-call** met parameters
- **Geen hardgecodeerde afstanden of hoeken** - alles parametrisch
- **JSON-gebaseerde commando's** via ROS topic `/fsm_commands`
- **Feedback op `/fsm_feedback`** voor elke actie
- **Backward compatibility** met legacy string commando's
- **Minimale FSM** met slechts 3 staten
- **Automatisch opstaan** als robot zit en actie niet 'sit' is

## Ondersteunde Acties

### Basis Beweging
```json
{"op": "stand", "height": 0.0}
{"op": "sit"}
{"op": "move_relative", "x": 1.2, "y": 0.0, "yaw": 1.570796, "timeout": 12}
```

### Arm Controle
```json
{"op": "arm_ready"}
{"op": "arm_stow"}
{"op": "aim_hand", "x": 0.75, "y": 0.0, "z": 0.55}
```

### Gripper
```json
{"op": "gripper", "mode": "open"}
{"op": "gripper", "mode": "close", "fraction": 0.5}
```

### Detectie en Picking
```json
{"op": "detect_label", "label": "bottle", "conf": 0.25}
{"op": "pick_from_pixel", "image_source": "hand_color_image", "x": 960, "y": 540}
{"op": "grasp_body_point", "x": 0.45, "y": 0.0, "z": 0.20}
{"op": "pick_apriltag", "tag_id": null, "z_offset": 0.20, "approach": 0.12, "lift": 0.15}
```

### Utility
```json
{"op": "sleep", "seconds": 2.0}
```

## Installatie en Gebruik

### 1. Start de FSM Node
```bash
# Terminal 1: Start de FSM node
rosrun spot_fsm_control fsm_node.py
```

### 2. Test JSON Commando's
```bash
# Terminal 2: Publiceer commando's
rostopic pub /fsm_commands std_msgs/String '{"op":"stand","height":0.0}'
rostopic pub /fsm_commands std_msgs/String '{"op":"move_relative","x":1.0,"y":0.0,"yaw":0.0}'
```

### 3. Gebruik Natural Language Control
```bash
# Terminal 3: Start NL control
rosrun spot_fsm_control natural_language_control.py
```

Voorbeelden van NL commando's:
- `loop 1.0 meter`
- `draai 90 graden links`
- `sta op hoog`
- `ga zitten`
- `richt hand naar beneden op 0.75 0.0 0.55`
- `pak bottle`
- `pak apriltag` of `pak tag 0`

### 4. Run Tests
```bash
# Terminal 4: Run test script
python test_v2_implementation.py
```

## Feedback Schema

Elke actie resulteert in feedback op `/fsm_feedback`:

**Success:**
```json
{"status": "ok", "op": "move_relative", "timestamp": 1234567890.123}
```

**Error:**
```json
{"status": "error", "op": "detect_label", "msg": "no detection", "timestamp": 1234567890.123}
```

## FSM Staten

1. **sit** - Robot zit
2. **stand** - Robot staat (neutrale staat)
3. **action** - Robot voert actie uit

Na elke actie keert de FSM terug naar `stand` (behalve bij `sit` actie).

## Detectie Functionaliteit

De `detect_label` actie:
- Gebruikt Ultralytics YOLO als beschikbaar
- Slaat laatste detection op voor `pick_from_pixel`
- Retourneert pixel coördinaten, bbox, label en confidence

## Veiligheid

- **Geen harde limieten** op afstanden of hoeken
- **Yaw normalisatie** naar [-π, π]
- **Gripper fraction clamping** tussen 0.0 en 1.0
- **Timeout handling** voor alle blokkerende operaties
- **Graceful shutdown** met arm stow en sit

## Backward Compatibility

Legacy string commando's worden automatisch gemapt naar JSON:
- `"stand"` → `{"op": "stand", "height": 0.0}`
- `"sit"` → `{"op": "sit"}`
- `"arm_ready"` → `{"op": "arm_ready"}`
- `"arm_stow"` → `{"op": "arm_stow"}`

## Troubleshooting

### Robot niet verbinden
- Controleer IP adres in `fsm_node.py` (default: 192.168.80.3)
- Zorg dat robot powered on is
- Controleer netwerk verbinding

### Commando's niet verwerkt
- Controleer of FSM node draait
- Controleer JSON syntax
- Kijk naar feedback op `/fsm_feedback`

### Detectie werkt niet
- Installeer Ultralytics: `pip install ultralytics`
- Controleer of hand camera beschikbaar is
- Verlaag confidence threshold

## Voorbeeld End-to-End Workflow

1. **Start robot:**
   ```bash
   rosrun spot_fsm_control fsm_node.py
   ```

2. **Sta op:**
   ```bash
   rostopic pub /fsm_commands std_msgs/String '{"op":"stand","height":0.0}'
   ```

3. **Loop vooruit:**
   ```bash
   rostopic pub /fsm_commands std_msgs/String '{"op":"move_relative","x":1.0,"y":0.0,"yaw":0.0}'
   ```

4. **Richt hand:**
   ```bash
   rostopic pub /fsm_commands std_msgs/String '{"op":"aim_hand","x":0.75,"y":0.0,"z":0.55}'
   ```

5. **Zoek object:**
   ```bash
   rostopic pub /fsm_commands std_msgs/String '{"op":"detect_label","label":"bottle"}'
   ```

6. **Pak object:**
   ```bash
   rostopic pub /fsm_commands std_msgs/String '{"op":"pick_from_pixel","image_source":"hand_color_image","x":960,"y":540}'
   ```

7. **Ga zitten:**
   ```bash
   rostopic pub /fsm_commands std_msgs/String '{"op":"sit"}'
   ```

## Ontwikkeling

De code is modulair opgezet voor eenvoudige uitbreiding:
- Nieuwe acties toevoegen in `_execute_action()` in FSM
- Nieuwe API calls in `SpotControlInterface`
- Nieuwe NL patronen in `parse_command()` in NL control
