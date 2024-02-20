#include <Wire.h>
#include <OLED_I2C.h>
#include <MS5837.h>


MS5837 sens;
OLED oled(SDA, SCL, 8);

extern uint8_t letters[];
extern uint8_t numbers[];


float depth, tempr;
void read_sens() {
	sens.read();
	
	depth = sens.depth();
	tempr = sens.temperature();
}


void display_values(float _depth, float _tempr) {
	oled.setFont(letters);
	oled.print("D:", LEFT, 0);
	oled.print("T:", LEFT, 40);

	oled.setFont(numbers);
	oled.printNumF(_depth, 2, CENTER, 0);
	oled.printNumF(_tempr, 2, CENTER, 40);

	oled.update();
}


void setup() {
	oled.begin();
	Wire.begin();
	Serial.begin();

	while(!sens.init()) {}
	sens.setModel(MS5837::MS5837_30BA);
	sens.setFluidDensity(997);	
}









