#include <Wire.h>
#include <OLED_I2C.h>
#include <MS5837.h>
#include <Adafruit_NeoPixel.h>

#define pixels_cnt 6
#define pixels_pin 2
#define pixels_brightness 10
#define oled_brightness 10
#define depth_step 0.25

double ms_offset;

MS5837 sens;
OLED oled(SDA, SCL);
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(pixels_cnt, pixels_pin, NEO_GRB + NEO_KHZ800);

extern uint8_t BigNumbers[];
extern uint8_t SmallFont[];

const uint32_t pixels_color[4] = {
    pixels.Color(0, 100, 0),   // 0 -> Green
    pixels.Color(100, 100, 0), // 1 -> Yellow
    pixels.Color(100, 0, 0),   // 2 -> Red
    pixels.Color(0, 0, 0)      // 3 -> Off
};
double depth, tempr;
bool in_danger, exit_danger_fl = true;
uint8_t depth_stage;
uint64_t danger_tmr; 

void read_sens() {
	sens.read();
	
	depth = max(sens.depth()-ms_offset, 0);
  	tempr = sens.temperature();

    depth_stage = int(max((float)depth, 0.0f) / depth_step);

    if(depth > 1.5) {
        if(!in_danger) danger_tmr = millis();

        in_danger = true;
    } else if(danger_tmr + 4000 < millis()) in_danger = false;
}
void for_dataset() {
    depth = (float)random(1000, 9999)/100;
    tempr = (float)random(100, 999)/100;
}

void display_values(double _depth, double _tempr, bool serial = 0) {
    oled.setBrightness(oled_brightness);
  	oled.setFont(SmallFont);
	oled.print("D:", LEFT, 0);
    oled.print("T:", LEFT, 40);

	oled.setFont(BigNumbers);
	oled.printNumF(_depth, 2, CENTER, 0);
  	oled.printNumF(_tempr, 1, CENTER, 40);

	oled.update();
    if(serial) {
        Serial.println("D: " + String(_depth));
        Serial.println("T: " + String(_tempr));
        Serial.println();
    }
}
void display_danger() {
    oled.setBrightness(oled_brightness);
    oled.setFont(SmallFont);
    oled.print("DANGER", 50, 20);
    oled.update();
}

void led_indicate(uint8_t ind) {
    for(int i=0; i<min(ind, pixels_cnt); ++i) 
        pixels.setPixelColor(pixels_cnt-i-1, pixels_color[(int)(ind-1)/2]);

    pixels.show();

}
void led_in_danger() {
    if(depth > 1.0f) {
        led_indicate(6);
        delay(200);

        for(int i=0; i<6; ++i) 
            pixels.setPixelColor(pixels_cnt-i-1, pixels_color[3]);

        pixels.show();  
        delay(200);
    } else 
        led_indicate(depth);
}


void clear() {
    oled.clrScr();
    pixels.clear();
}


void setup() {
    oled.begin();
    pixels.begin();
    Wire.begin();
    Serial.begin(115200);

    while(!sens.init()) {}
    sens.setModel(MS5837::MS5837_30BA);
    sens.setFluidDensity(997);	

    read_sens();
    ms_offset = depth;

    oled.setBrightness(oled_brightness);
    pixels.setBrightness(pixels_brightness);
}
void loop() {
    read_sens();   

    switch (in_danger) {
        case true:
            display_danger();
            led_in_danger();
            break;
        
        case false:
            led_indicate(depth_stage);
            display_values(depth, tempr);
            break;
    }

    delay(50);
    clear();
}