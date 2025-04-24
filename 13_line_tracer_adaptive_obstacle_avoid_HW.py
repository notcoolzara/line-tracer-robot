from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# === Connect to CoppeliaSim ===
client = RemoteAPIClient()
sim = client.require('sim')

# === Object Handles ===
sensors = {
    "left": sim.getObject('/LeftSensor'),
    "right": sim.getObject('/RightSensor'),
    "mid": sim.getObject('/MiddleSensor')
}
motors = {
    "left": sim.getObject('/DynamicLeftJoint'),
    "right": sim.getObject('/DynamicRightJoint')
}
vision_sensor = sim.getObject('/Vision_sensor')
proximity_sensor = sim.getObject('/Proximity_sensor')
linetracer = sim.getObject('/LineTracer')

# === PID and Simulation Parameters ===
dmax = 0.35            # Max sensor distance if nothing is detected
Kp = 0.23             # Stronger immediate correction
Kd = 0.04              # Dampens sudden error change
Ki = 0.004   
v_base = 1 
e_old = 0              # Previous error
e_integral = 0         # Cumulative error for I term
d_obstacle = 0.22      # Obstacle detection threshold

# === Helper Functions ===
def get_image(sensor):
    img, res = sim.getVisionSensorImg(sensor)
    if img:
        img = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return None

def sensor_ir(sensor):
    # Basic IR thresholding (value > 40 = line detected)
    img, _ = sim.getVisionSensorImg(sensor)
    return np.frombuffer(img, dtype=np.uint8)[0] > 40 if img else False

def get_distance(sensor):
    # Get distance from proximity sensor
    state, point, *_ = sim.readProximitySensor(sensor)
    return np.linalg.norm(point) if state else dmax

# === Main Logic ===
def execute():
    global e_old, e_integral
    state = "LINE_FOLLOWING"
    vL_list, vR_list, px, py = [], [], [], []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)

    start_pos = sim.getObjectPosition(linetracer, -1)
    lap_threshold = 0.2
    lap_start_time = time.time()
    lap_count = 0
    lap_times = []
    in_zone = True

    while sim.getSimulationState() != sim.simulation_stopped:
        # Track robot position for laps
        pos = sim.getObjectPosition(linetracer, -1)
        px.append(pos[0])
        py.append(pos[1])
        d_from_start = np.linalg.norm(np.array(pos[:2]) - np.array(start_pos[:2]))

        if in_zone and d_from_start > lap_threshold:
            in_zone = False
        if not in_zone and d_from_start < lap_threshold:
            lap_time = time.time() - lap_start_time
            lap_count += 1
            lap_times.append(lap_time)
            lap_start_time = time.time()
            in_zone = True
            print(f"[INFO] Lap {lap_count} completed in {lap_time:.2f} seconds.")
            if lap_count == 2:
                break

        # Read sensors
        d = get_distance(proximity_sensor)
        imgL, imgM, imgR = sensor_ir(sensors["left"]), sensor_ir(sensors["mid"]), sensor_ir(sensors["right"])

        # === LINE FOLLOWING ===
        if state == "LINE_FOLLOWING":
            if d < d_obstacle:
                print("[STATE] Obstacle detected. Switching to OBSTACLE_AVOIDANCE.")
                state = "OBSTACLE_AVOIDANCE"
                continue

            # Determine error based on IR sensors
            if imgL and not imgM and not imgR:
                e_new = -1
            elif imgL and imgM and not imgR:
                e_new = -0.5
            elif imgR and not imgM and not imgL:
                e_new = 1
            elif imgR and imgM and not imgL:
                e_new = 0.5
            elif imgM and not imgL and not imgR:
                e_new = 0
            else:
                e_new = 0  # Default

            # Adaptive PID based on error magnitude
            adaptive_Kp = Kp * 1.5 if abs(e_new) > 0 else Kp
            adaptive_Kd = Kd * 1.5 if abs(e_new) > 0 else Kd

            # PID correction
            e_integral += e_new
            correction = adaptive_Kp * e_new + adaptive_Kd * (e_new - e_old) + Ki * e_integral
            e_old = e_new
            

                        # Adjust speed during sharp turns
            if abs(e_new) > 0.5:         # Sharp turn detected (you can tune 0.5)
                v = 0.65                 # Lower speed for better turning
            else:
                v = v_base               # Normal speed
            
            # Apply motor speeds with adjusted base velocity
            vL = v - correction
            vR = v + correction
            sim.setJointTargetVelocity(motors["left"], vL)
            sim.setJointTargetVelocity(motors["right"], vR)


            vL_list.append(vL)
            vR_list.append(vR)

        # === OBSTACLE AVOIDANCE ===
        elif state == "OBSTACLE_AVOIDANCE":
            print("[STATE] Avoiding obstacle: turn left then move forward.")
            sim.setJointTargetVelocity(motors["left"], 1.2)
            sim.setJointTargetVelocity(motors["right"], -1.2)
            time.sleep(0.4)

            sim.setJointTargetVelocity(motors["left"], 1.2)
            sim.setJointTargetVelocity(motors["right"], 1.2)
            time.sleep(0.3)

            state = "LINE_SEARCH"
            search_start_time = time.time()

        # === LINE SEARCH ===
        elif state == "LINE_SEARCH":
            print("[STATE] Searching for line...")
            search_duration = 5.0

            while time.time() - search_start_time < search_duration:
                d = get_distance(proximity_sensor)
                imgL, imgM, imgR = sensor_ir(sensors["left"]), sensor_ir(sensors["mid"]), sensor_ir(sensors["right"])
                found_line = not((imgL or imgR) and imgM)

                sim.setJointTargetVelocity(motors["left"], -0.5)
                sim.setJointTargetVelocity(motors["right"], 0.5)

                if d < d_obstacle:
                    print("[STATE] Obstacle detected during search. Switching back.")
                    state = "OBSTACLE_AVOIDANCE"
                    break
                elif found_line:
                    print("[STATE] Line detected. Moving to LINE_ALIGNMENT.")
                    state = "LINE_ALIGNMENT"
                    break
                time.sleep(0.001)
            else:
                print("[STATE] Line search failed. Reversing spin.")
                sim.setJointTargetVelocity(motors["left"], 0.5)
                sim.setJointTargetVelocity(motors["right"], -0.5)
                time.sleep(0.5)
                state = "LINE_SEARCH"
                search_start_time = time.time()

        # === LINE ALIGNMENT ===
        elif state == "LINE_ALIGNMENT":
            print("[STATE] Aligning with line...")
            found_line = 1
            while found_line:
                imgL, imgM, imgR = sensor_ir(sensors["left"]), sensor_ir(sensors["mid"]), sensor_ir(sensors["right"])
                if imgL and not imgM and not imgR:
                    while True:
                        sim.setJointTargetVelocity(motors["left"], 0.5)
                        sim.setJointTargetVelocity(motors["right"], -0.4)
                        time.sleep(0.04)
                        imgL, imgM, imgR = sensor_ir(sensors["left"]), sensor_ir(sensors["mid"]), sensor_ir(sensors["right"])
                        if imgR and not imgM and not imgL:
                            break
                    print("[ALIGN] Adjusting left.")
                elif imgR and not imgM and not imgL:
                    sim.setJointTargetVelocity(motors["left"], 0.3)
                    sim.setJointTargetVelocity(motors["right"], 0.1)
                    print("[ALIGN] Adjusting right.")
                elif imgL and imgM:
                    sim.setJointTargetVelocity(motors["left"], 0.2)
                    sim.setJointTargetVelocity(motors["right"], 0.1)
                    print("[ALIGN] Slight left.")
                elif imgR and imgM:
                    sim.setJointTargetVelocity(motors["left"], 0.2)
                    sim.setJointTargetVelocity(motors["right"], 0.1)
                    print("[ALIGN] Slight right.")
                time.sleep(0.05)
                imgL, imgM, imgR = sensor_ir(sensors["left"]), sensor_ir(sensors["mid"]), sensor_ir(sensors["right"])
                found_line = not((imgL or imgR) and imgM)

            print("[STATE] Line alignment complete. Resuming line following.")
            state = "LINE_FOLLOWING"
            continue

        # === Plot Path and Speeds ===
        ax1.clear()
        ax1.scatter(py, px)
        ax1.set_xlabel("X axis")
        ax1.set_ylabel("Y axis")
        ax1.set_title("Robot Path")
        ax1.grid()

        ax2.clear()
        ax2.plot(vL_list, label="Left Motor")
        ax2.plot(vR_list, label="Right Motor")
        ax2.set_title("Motor Speeds")
        ax2.legend()
        ax2.grid()

        plt.pause(0.001)

    cv2.destroyAllWindows()
    print("\n=== Lap Times ===")
    for i, t in enumerate(lap_times):
        print(f"Lap {i + 1}: {t:.2f} seconds")

# === Start Simulation ===
def start_simulation():
    sim.startSimulation()
    execute()
    sim.stopSimulation()

if __name__ == '__main__':
    start_simulation()