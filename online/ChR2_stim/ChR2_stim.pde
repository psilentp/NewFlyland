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

void wb_triggered(){
  if(go_val){
    delay(delay_val);
    digitalWrite(led_pin,LOW);
    digitalWrite(sync_pin,HIGH);
    delay(led_pulse);
  }
}

void tmr_triggered(){
  if(go_val){
    digitalWrite(led_pin,LOW);
    digitalWrite(sync_pin,HIGH);
    delay(delay_val);
  }
}


void set_state(){
  x_state = (x_state*0.9 + (float(analogRead(x_in))/x_max)*0.1);
  y_state = (y_state*0.9 + (float(analogRead(y_in))/y_max)*0.1);
  delay_val = x_state*10.0;
  go_val = (y_state > 0.1);
}
