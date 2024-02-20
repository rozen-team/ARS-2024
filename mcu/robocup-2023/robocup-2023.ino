#include <OLED_I2C.h>
OLED myOLED(SDA, SCL);
extern uint8_t SmallFont[];


void setup() {
     myOLED.begin();
    myOLED.setFont(SmallFont); // SmallFont

     Serial.begin(115200);
}

void loop() {
  Serial.println("Hohfwojwijfp");

  myOLED.print("SIGMA", CENTER, 24);
}
