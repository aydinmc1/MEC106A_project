# Current state of the play operation state machine
std_msgs/Header header

uint8 IDLE        = 0
uint8 HOMING      = 1
uint8 POSITIONING = 2
uint8 OPERATING   = 3
uint8 RETRACTING  = 4
uint8 COMPLETE    = 5
uint8 ERROR       = 6
uint8 PAUSED      = 7

uint8 state
string state_label
string description
float32 progress_percent   # 0.0 - 100.0
