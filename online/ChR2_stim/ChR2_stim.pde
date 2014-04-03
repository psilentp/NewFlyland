int x_in = 3;
int y_in = 4;
int trigger_pin = 22;
int led_pin = 27;
int sync_pin = 25;
//int stim_pin = 27;
float x_state = 0;
float y_state = 0;

float delay_val = 100;
int go_val = true;

float x_max = 3867;
float y_max = 3910;

int stim_rate = 1000*100; //in microseconds

int led_pulse = 1;

HardwareTimer timer(2);

void setup() {
    pinMode(led_pin, OUTPUT_OPEN_DRAIN);
    pinMode(sync_pin, OUTPUT);
    pinMode(x_in, INPUT_ANALOG);
    pinMode(y_in, INPUT_ANALOG);
    ///////////////////////
    ///////////////////////
    ///////////////////////
    //setup_for_wbstim();
    setup_for_timed(); 
    ///////////////////////
    ///////////////////////
    ///////////////////////  
}

void setup_for_timed(){
   timer.pause();
   timer.setPeriod(stim_rate);
   timer.setChannel1Mode(TIMER_OUTPUT_COMPARE);
   timer.setCompare(TIMER_CH1, 1);
   timer.attachCompare1Interrupt(tmr_triggered);
   timer.refresh();
   timer.resume();
}

void setup_for_wbstim(){
  attachInterrupt(trigger_pin,wb_triggered,FALLING);
}

void loop() {
    set_state(); 
    digitalWrite(led_pin,HIGH);
    digitalWrite(sync_pin,LOW);
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
