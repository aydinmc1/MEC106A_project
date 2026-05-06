# Summary of camera intrinsics and runtime stats
std_msgs/Header header

# Intrinsics
float64 fx
float64 fy
float64 cx
float64 cy
float64[5] distortion_coeffs

# Runtime
float32 fps_measured
int32 frame_count
bool is_healthy
string status_message
