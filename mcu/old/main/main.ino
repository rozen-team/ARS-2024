#include <Wire.h>
#include <OLED_I2C.h>
#include <U8glib.h>
#include <MS5837.h>
#include <Adafruit_NeoPixel.h>

#define pixels_cnt 6
#define pixels_pin 2
#define depth_step 0.25

MS5837 sens;
OLED oled(SDA, SCL);
U8GLIB_SSD1306_128X64 u8g(U8G_I2C_OPT_NO_ACK);
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(pixels_cnt, pixels_pin, NEO_GRB + NEO_KHZ800);

extern uint8_t BigNumbers[];
extern uint8_t SmallFont[];

const uint8_t pixels_color[3] = {
  pixels.Color(0, 255, 0),    // 0 -> Green
  pixels.Color(255, 255, 0),  // 1 -> Yellow
  pixels.Color(255, 0, 0),    // 2 -> Red
};
float depth, tempr;
bool in_danger;
uint8_t depth_stage;


void read_sens() {
  sens.read();

  depth = sens.depth();
  tempr = sens.temperature();

  depth_stage = (int)depth / depth_step;
  if (depth_stage > 6)
    in_danger = true;
}


void display_values(float _depth, float _tempr, bool serial = 0) {
  oled.setFont(SmallFont);
  oled.print("D:", LEFT, 0);
  oled.print("T:", LEFT, 40);

  oled.setFont(BigNumbers);
  oled.printNumF(_depth, 2, CENTER, 0);
  oled.printNumF(_tempr, 1, CENTER, 40);

  oled.update();
  Serial.println("D: " + String(_depth));
  Serial.println("T: " + String(_tempr));
  Serial.println();
}
void display_danger() {
  u8g.firstPage();

  u8g.setFont(u8g_font_fub25n);
  u8g.drawStr(0, 22, "DANGER");

  delay(4000);
  in_danger = false;

  u8g.nextPage();
}
void set_brightness(int _brightness) {
  oled.setBrightness(_brightness);
}


void led_indicate(int ind) {
  for (int i = 0; i < min(ind, pixels_cnt); ++i)
    pixels.setPixelColor(pixels_cnt - i - 1, pixels_color[(int)i / 2]);

  pixels.show();
}


void for_dataset() {
  depth = (float)random(1000, 9999) / 100;
  tempr = (float)random(100, 999) / 100;
}


void setup() {
  oled.begin();
  pixels.begin();
  Wire.begin();
  Serial.begin(115200);

  while (!sens.init()) {}
  sens.setModel(MS5837::MS5837_30BA);
  sens.setFluidDensity(997);

  if (u8g.getMode() == U8G_MODE_R3G3B2) {
    u8g.setColorIndex(255);  // white
  } else if (u8g.getMode() == U8G_MODE_GRAY2BIT) {
    u8g.setColorIndex(3);  // max intensity
  } else if (u8g.getMode() == U8G_MODE_BW) {
    u8g.setColorIndex(1);  // pixel on
  } else if (u8g.getMode() == U8G_MODE_HICOLOR) {
    u8g.setHiColorByRGB(255, 255, 255);
  }
}


void loop() {
  read_sens();

  if (in_danger)
    display_danger();

  else {
    display_values(depth, tempr, 1);
    led_indicate(depth_stage);
  }

  oled.clrScr();
  pixels.clear();
  delay(50);
}
