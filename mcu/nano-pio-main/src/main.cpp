#include <Wire.h>
#include <OLED_I2C.h>
#include <U8glib.h>
#include <MS5837.h>
#include <Adafruit_NeoPixel.h>

#define pixels_cnt 6
#define pixels_pin 2
#define pixels_brightness 10
#define oled_brightness 10
#define depth_step 0.25
#define ms_offset 0.48


MS5837 sens;
OLED oled(SDA, SCL);
U8GLIB_SSD1306_128X64 u8g(U8G_I2C_OPT_NO_ACK);
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(pixels_cnt, pixels_pin, NEO_GRB + NEO_KHZ800);

extern uint8_t BigNumbers[];
extern uint8_t SmallFont[];

const uint32_t pixels_color[4] = {
    pixels.Color(0, 100, 0),   // 0 -> Green
    pixels.Color(100, 100, 0), // 1 -> Yellow
    pixels.Color(100, 0, 0),   // 2 -> Red
    pixels.Color(0, 0, 0)
};
double depth, tempr;
bool in_danger;
uint8_t depth_stage;


void read_sens() {
	sens.read();
	
	depth = sens.depth() - ms_offset;
  	tempr = sens.temperature();

    depth_stage = int(max((float)depth, 0.0f) / depth_step);
    depth_stage > 6 || depth > 1.5 ? in_danger = true : in_danger = false;
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
void display_danger_u8g() {
    u8g.firstPage();

    u8g.setFont(u8g_font_unifont);
    u8g.drawStr(0, 22, "DANGER");
  
    delay(4000); in_danger = false;

    u8g.nextPage();
}

void led_indicate(uint8_t ind) {
    for(int i=0; i<min(ind, pixels_cnt); ++i) 
        pixels.setPixelColor(pixels_cnt-i-1, pixels_color[(int)(ind-1)/2]);

    pixels.show();

}
void led_in_danger() {
    led_indicate(6);
    delay(200);

    for(int i=0; i<6; ++i) 
        pixels.setPixelColor(pixels_cnt-i-1, pixels_color[3]);

    pixels.show();  
    delay(200);
}

void ddanger() {

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

    // if ( u8g.getMode() == U8G_MODE_R3G3B2 ) {
    //     u8g.setColorIndex(255);     // white
    // }
    // else if ( u8g.getMode() == U8G_MODE_GRAY2BIT ) {
    //     u8g.setColorIndex(3);         // max intensity
    // }
    // else if ( u8g.getMode() == U8G_MODE_BW ) {
    //     u8g.setColorIndex(1);         // pixel on
    // }
    // else if ( u8g.getMode() == U8G_MODE_HICOLOR ) {
    //     u8g.setHiColorByRGB(255,255,255);
    // }

    oled.setBrightness(oled_brightness);
    pixels.setBrightness(pixels_brightness);
}
void loop() {
    read_sens();   

    if(in_danger) {
        clear();                

        display_danger();
        auto tmr = millis();
        while(tmr+4000 > millis()) { led_in_danger(); }


    } else {
        led_indicate(depth_stage);
        display_values(depth, tempr);
    }

    delay(50);
    clear();
}