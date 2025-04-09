#include <Servo.h>

#define DIR_PIN_SYNC 8   // Direction pin for synced motors (Y-axis)
#define PUL_PIN_SYNC 9   // Pulse (Step) pin for synced motors
#define ENA_PIN_SYNC 10  // Enable pin for synced motors
#define LIMIT_SWITCH_SYNC 7  // Limit switch NO pin for synced motors

#define DIR_PIN_SINGLE 4   // Direction pin for single motor (X-axis)
#define PUL_PIN_SINGLE 5   // Pulse (Step) pin for single motor
#define ENA_PIN_SINGLE 6   // Enable pin for single motor
#define LIMIT_SWITCH_SINGLE 3  // Limit switch NO pin for single motor

#define SERVO_PIN 11       // Servo motor pin
#define CLAW_OPEN 0        // Open claw position
#define CLAW_CLOSED 40     // Closed claw position

#define POSITION_SYNC_1 1000  // Bottom row (safe move)
#define POSITION_SYNC_2 25000 // Top row (grabbing position)

#define T1Move 500   // Tool 4 location (furthest right)
#define T2Move 8500  // Tool 3 location
#define T3Move 16500 // Tool 2 location
#define T4Move 24500 // Tool 1 location
#define T5Move 32500 // Second dispense location (unused)
#define T6Move 40500 // Leftmost dispense location (final drop-off)

long currentPositionSingle = T6Move;
long currentPositionSync = POSITION_SYNC_1;

int minDelay = 100;
int maxDelay = 300;
int accelSteps = 300;

Servo clawServo;

void setup() {
    Serial.begin(9600);
    pinMode(DIR_PIN_SYNC, OUTPUT);
    pinMode(PUL_PIN_SYNC, OUTPUT);
    pinMode(ENA_PIN_SYNC, OUTPUT);
    pinMode(LIMIT_SWITCH_SYNC, INPUT_PULLUP);
    
    pinMode(DIR_PIN_SINGLE, OUTPUT);
    pinMode(PUL_PIN_SINGLE, OUTPUT);
    pinMode(ENA_PIN_SINGLE, OUTPUT);
    pinMode(LIMIT_SWITCH_SINGLE, INPUT_PULLUP);
    
    digitalWrite(ENA_PIN_SYNC, LOW);
    digitalWrite(ENA_PIN_SINGLE, LOW);
    
    clawServo.attach(SERVO_PIN);
    clawServo.write(CLAW_CLOSED); // Keep claw closed initially
    
    Serial.println("Homing... Moving to limit switches.");
    homeWithLimitSwitch(DIR_PIN_SYNC, PUL_PIN_SYNC, LIMIT_SWITCH_SYNC, currentPositionSync);
    homeWithLimitSwitch(DIR_PIN_SINGLE, PUL_PIN_SINGLE, LIMIT_SWITCH_SINGLE, currentPositionSingle);
    Serial.println("Homing complete. Waiting for commands (D1-D4, R1-R4)");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        long toolPosition = 0;
        
        if (input == "D1") toolPosition = T4Move;
        else if (input == "D2") toolPosition = T3Move;
        else if (input == "D3") toolPosition = T2Move;
        else if (input == "D4") toolPosition = T1Move;
        else if (input == "R1") toolPosition = T4Move; // Retrieve tool 1
        else if (input == "R2") toolPosition = T3Move; // Retrieve tool 2
        else if (input == "R3") toolPosition = T2Move; // Retrieve tool 3
        else if (input == "R4") toolPosition = T1Move; // Retrieve tool 4
        else {
            Serial.println("Invalid input. Use D1-D4 or R1-R4.");
            return;
        }

        // Ensure Y-axis is in bottom row
        moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_1, currentPositionSync);
        
        if (input.charAt(0) == 'D') {
            // Dispense tool
            moveToPosition(DIR_PIN_SINGLE, PUL_PIN_SINGLE, toolPosition, currentPositionSingle);
            // Open gripper
            clawServo.write(CLAW_OPEN);
            delay(500);

            // Move Y to top row
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_2, currentPositionSync);

            // Close gripper (grabbing tool)
            clawServo.write(CLAW_CLOSED);
            delay(500);

            // Move Y back to bottom row
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_1, currentPositionSync);

            // Move X to leftmost dispense location (T6Move)
            moveToPosition(DIR_PIN_SINGLE, PUL_PIN_SINGLE, T6Move, currentPositionSingle);

            // Move Y to top row to release tool
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_2, currentPositionSync);

            // Open gripper (releasing tool)
            clawServo.write(CLAW_OPEN);
            delay(500);

            // Move Y back to bottom row
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_1, currentPositionSync);
            
            // Move X back to home position (T6Move)
            moveToPosition(DIR_PIN_SINGLE, PUL_PIN_SINGLE, T6Move, currentPositionSingle);

            Serial.println("Tool dispensed. Waiting for next command (D1-D4 or R1-R4)");
        } else if (input.charAt(0) == 'R') {
            // Retrieve tool
            // Move to leftmost dispense location (T6Move)

    
            moveToPosition(DIR_PIN_SINGLE, PUL_PIN_SINGLE, T6Move, currentPositionSingle);
            
            // Ensure gripper is open
            clawServo.write(CLAW_OPEN);
            delay(500);

            // Move Y to top row
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_2, currentPositionSync);

            // Close gripper (grabbing tool)
            clawServo.write(CLAW_CLOSED);
            delay(500);

            // Move Y back to bottom row
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_1, currentPositionSync);

            // Move to correct tool position
            moveToPosition(DIR_PIN_SINGLE, PUL_PIN_SINGLE, toolPosition, currentPositionSingle);

            // Move Y to top row to release the tool
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_2, currentPositionSync);

            // Open gripper (releasing tool)
            clawServo.write(CLAW_OPEN);
            delay(500);

            // Move Y back to bottom row
            moveToPosition(DIR_PIN_SYNC, PUL_PIN_SYNC, POSITION_SYNC_1, currentPositionSync);

            // Move X back to home position (T6Move)
            moveToPosition(DIR_PIN_SINGLE, PUL_PIN_SINGLE, T6Move, currentPositionSingle);

            Serial.println("Tool retrieved. Waiting for next command (D1-D4, R1-R4)");
        }
    }
}

void moveToPosition(int dirPin, int pulPin, long targetPosition, long &currentPosition) {
    long stepsToMove = targetPosition - currentPosition;
    if (stepsToMove == 0) {
        Serial.println("Already at this position.");
        return;
    }
    bool direction = (stepsToMove > 0);
    Serial.print("Moving ");
    Serial.println(direction ? "forward..." : "backward...");
    moveStepper(dirPin, pulPin, direction, abs(stepsToMove));
    currentPosition = targetPosition;
}

void homeWithLimitSwitch(int dirPin, int pulPin, int limitSwitchPin, long &currentPosition) {
    Serial.println("Moving to home...");
    digitalWrite(dirPin, LOW);
    while (digitalRead(limitSwitchPin) == HIGH) {
        digitalWrite(pulPin, HIGH);
        delayMicroseconds(500);
        digitalWrite(pulPin, LOW);
        delayMicroseconds(500);
    }
    currentPosition = 0;
}

void moveStepper(int dirPin, int pulPin, bool direction, long steps) {
    digitalWrite(dirPin, direction ? HIGH : LOW);
    int delayTime = maxDelay;
    for (long i = 0; i < steps; i++) {
        digitalWrite(pulPin, HIGH);
        delayMicroseconds(delayTime);
        digitalWrite(pulPin, LOW);
        delayMicroseconds(delayTime);
        if (i < accelSteps && delayTime > minDelay) {
            delayTime -= 2;
        }
        if (i > (steps - accelSteps) && delayTime < maxDelay) {
            delayTime += 2;
        }
    }
}
