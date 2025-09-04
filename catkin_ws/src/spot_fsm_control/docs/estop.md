## Emergency Stop (Estop) Commands

### Start normal ROS Launch
```bash
roslaunch spot_fsm_control spot_fsm_control.launch
```

### Trigger Estop (Cut)
```bash
rosservice call /estop/stop "{}"
```

### Release Estop (Allow)
```bash
rosservice call /estop/allow "{}"
```

### Settle Then Cut
```bash
rosservice call /estop/settle_then_cut "{}"
```
