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
double depth, prev_depth, tempr;
uint8_t depth_stage;
uint64_t danger_tmr; 
bool in_danger, in_normal = true, danger_blink, going_up;


void read_sens() {
	sens.read();
	prev_depth = depth;

	depth = max(0, sens.depth()-ms_offset);
  	tempr = sens.temperature();

    depth_stage = int(max((float)depth, 0.0f) / depth_step);

    if(depth > 1.5) {
        if(in_normal) danger_tmr = millis(), in_normal = false, danger_blink = true;

        if(!going_up) { if(depth < prev_depth) going_up = true; }
        else            if(depth > prev_depth) going_up = false;

        in_danger = false ? false : true; // change <going_up> to false/0 if u do not need new logic
    } else if(!in_normal) {
        if(danger_tmr + 1600 < millis()) 
        depth <= 1.5 ? depth <= 1.0 ? danger_blink = false,
                       in_danger = false, going_up = false, in_normal = true :
                       in_danger = false, going_up = false, in_normal = true :
                       in_danger = true,                    in_normal = false; 
    } else in_danger = false, in_normal = true;

    if(depth <  1.5 && danger_tmr + 1600 < millis()) in_normal = true;
    if(depth <= 1.0) danger_blink = false;
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
    // Serial.println((float)danger_tmr);
}

void led(uint8_t ind, uint32_t _pixels_color = pixels_color[(int)(depth_stage-1) / 2]) {
    for(int i=0; i<min(ind, pixels_cnt); ++i) 
        pixels.setPixelColor(pixels_cnt-i-1, _pixels_color);

    pixels.show();
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

    in_danger ? display_danger() : display_values(depth, tempr);
    danger_blink ? led(pixels_cnt, pixels_color[int((millis() - danger_tmr) / 200) % 2 ? 2 : 3]) : led(depth_stage);

    delay(50);
    clear();
}