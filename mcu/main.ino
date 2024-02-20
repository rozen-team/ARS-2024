#include <OLED_I2C.h>

OLED oled(SDA, SCL, 8);

extern uint8_t letters[];
extern uint8_t numbers[];


void display_values(float depth, float tempr) {
	oled.setFont(letters);
	oled.print("D:", LEFT, 0);
	oled.print("T:", LEFT, 40);

	oled.setFont(numbers);
	oled.printNumF(depth, 2, CENTER, 0);
	oled.printNumF(tempr, 2, CENTER, 40);

	oled.update();
}




void setup() {
	oled.begin();
}





