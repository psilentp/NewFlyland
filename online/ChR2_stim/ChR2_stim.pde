/*
 Blink

 Turns on the built-in LED on for one second, then off for one second,
 repeatedly.

 Ported to Maple from the Arduino example 27 May 2011
 By Marti Bolivar
*/
int go_pin = 30;
volatile int wb_count;

void setup() {
    // Set up the built-in LED pin as an output:
    pinMode(0, OUTPUT_OPEN_DRAIN);
    pinMode(19,OUTPUT);
    pinMode(go_pin,INPUT);
    attachInterrupt(31,triggered,FALLING);
}

void loop() {
    digitalWrite(0,HIGH);
    digitalWrite(19,LOW);
}

void triggered(){
  if(digitalRead(go_pin)){
    if(wb_count > 2){
      wb_count = 0;
      digitalWrite(0,LOW);
      digitalWrite(19,HIGH);
      delay(2);
    }
    else{
      wb_count++;
    }
  }
}
