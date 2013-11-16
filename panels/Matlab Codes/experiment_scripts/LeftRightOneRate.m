expan_pole = 1; %y value of expansion this maps to  the azmuthal focus of expansion
approach_time = 1; %function number of expansion rate 
fixation_pat = 1;
expan_pat = 2;
record_index = 1;
expan_update_freq = 90;

%% Closed loop
Panel_com('stop');pause(0.005);
Panel_com('set_pattern_id',fixation_pat);pause(0.005); %stripe fixation pattern
Panel_com('set_mode',[1 1]);pause(0.05); %set closed loop X closed loop Y
Panel_com('set_velfunc_id',[0 0]);pause(0.01); %use defaul  t function on x and y channel (is this nessessary?)
Panel_com('send_gain_bias',[30.0,0,0,0]);pause(0.005);%
Panel_com('set_position',[48 1]);pause(0.005); % start at close to fixation
Panel_com('start');
pause(5);

%for i = datasample(1:5,1:5;ones(1,5)*5,ones(1,5)*11],10,2,'replace',false)
%for i = datasample(1:5,1:5;ones(1,5)*5,ones(1,5)*11],10,2,'replace',false)
%for i = datasample([ones(1:10)*2;ones(1,5)*5,ones(1,5)*11],10,2,'replace',false)

%Five reps of one aproach rate, presented from both sides.
%approach rate index = 2 corresponds to 20ms approach rate

for rep = 1:5
    for i = datasample([ones(1,10)*2;ones(1,5)*5,ones(1,5)*11],10,2,'replace',false)
        approach_time = i(1);
        expan_pole = i(2);
        datarecord(record_index).approach_time = approach_time;
        datarecord(record_index).expan_pole = expan_pole;
        datarecord(record_index).start_time = now;
        
        %% Closed loop
        Panel_com('stop');pause(0.005);
        Panel_com('set_pattern_id',fixation_pat);pause(0.005); %stripe fixation pattern
        Panel_com('set_mode',[1 1]);pause(0.05); %set closed loop X closed loop Y
        Panel_com('set_velfunc_id',[0 0]);pause(0.01); %use default function on x and y channel (is this nessessary?)
        Panel_com('send_gain_bias',[30.0,0,0,0]);pause(0.005);%
        Panel_com('set_position',[48 1]);pause(0.005); % start at close to fixation
        Panel_com('start');
        pause(5);

        %% Start expansion
        Panel_com('stop');
        %%Panel_com('ident_compress_on');pause(0.005);
        Panel_com('set_pattern_id',expan_pat);pause(0.005); %set to expansion pattern 
        Panel_com('set_mode',[4 0]);pause(0.05); %set the x mode to function based position control mode
        Panel_com('send_gain_bias',[0,0,0,0]);pause(0.005); %set the gain and bias to zero for both X and Y
        Panel_com('set_position',[2 expan_pole]);pause(0.005); %set the x position at begining y posistion is an exp parameter
        Panel_com('set_posfunc_id',[1 approach_time]);pause(0.01); %position function to run on channel 1 (X) 
        Panel_com('set_funcx_freq', expan_update_freq);pause(0.1); %run at 500hz 2ms steps
        Panel_com('start');
        pause(1.2);
        record_index = record_index+1;
        %%Panel_com('ident_compress_off');pause(0.005);
        i
    end
end

%% Put the fly back into closed loop.
Panel_com('stop');pause(0.005);
Panel_com('set_pattern_id',fixation_pat);pause(0.005); %stripe fixation pattern
Panel_com('set_mode',[1 1]);pause(0.05); %set closed loop X closed loop Y
Panel_com('set_velfunc_id',[0 0]);pause(0.01); %use default function on x and y channel (is this nessessary?)
Panel_com('send_gain_bias',[30.0,0,0,0]);pause(0.005);%
Panel_com('set_position',[48 1]);pause(0.005); % start at close to fixation
Panel_com('start');
pause(10);
