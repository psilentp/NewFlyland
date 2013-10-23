/*
 Blink

 Turns on the built-in LED on for one second, then off for one second,
 repeatedly.

 Ported to Maple from the Arduino example 27 May 2011
 By Marti Bolivar
*/
int
void setup() {
    // Set up the built-in LED pin as an output:
    pinMode(0, OUTPUT_OPEN_DRAIN);
    pinMode(19,OUTPUT);
    attachInterrupt(31,triggered,FALLING);
}

void loop() {
    delay(5);
    digitalWrite(0,HIGH);
    digitalWrite(19,LOW);
}

void triggered(){
  digitalWrite(0,LOW);
  digitalWrite(19,HIGH);
}
