#include <OLED_I2C.h>

OLED  myOLED(SDA, SCL, 8);

extern uint8_t SmallFont[];
extern uint8_t MediumNumbers[];
extern uint8_t BigNumbers[];

void setup()
{
  myOLED.begin();
  myOLED.setFont(SmallFont);
}

void loop()
{
  for (int i=0; i<=10000; i++)
  {
    myOLED.setFont(SmallFont);
    myOLED.print("D:", LEFT, 0);
    myOLED.print("T:", LEFT, 40);
    
    myOLED.setFont(BigNumbers);
    myOLED.printNumF(float(i)/3, 2, CENTER, 0);
    myOLED.printNumF(float(i)/3, 2, CENTER, 40);    
    
    myOLED.update(); delay(100);
  }
  
  // myOLED.setFont(SmallFont);
  // myOLED.print("|", LEFT, 24);
  // myOLED.print("|", RIGHT, 24);
  // myOLED.update();
  // delay(500);
  // for (int i=0; i<19; i++)
  // {
  //   myOLED.print("\\", 7+(i*6), 24);
  //   myOLED.update();
  //   delay(250);
  // }
  myOLED.clrScr();
}


