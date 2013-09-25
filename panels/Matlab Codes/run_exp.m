%Panel_com('set_ao',[1,0])
Panel_com('stop');;pause(0.005);
Panel_com('set_pattern_id',1);pause(0.005);
Panel_com('set_mode',[1 1]);pause(0.05);
Panel_com('set_velfunc_id',[0 0]);pause(0.01);
Panel_com('send_gain_bias',[60.0,0,0,0]);pause(0.005);
Panel_com('set_position',[2 1]);pause(0.005);
Panel_com('start');
pause(1);
Panel_com('stop');
Panel_com('set_mode',[4 0]);pause(0.05);
Panel_com('send_gain_bias',[0,0,0,0]);pause(0.005);
Panel_com('set_position',[1 2]);pause(0.005)
Panel_com('set_posfunc_id',[1 1]);pause(0.01);
Panel_com('set_funcx_freq', 500);pause(0.1);
%Panel_com('laser_on')
Panel_com('start');
pause(2);
Panel_com('stop');
Panel_com('set_pattern_id',1);pause(0.005);
Panel_com('set_mode',[1 1]);pause(0.05);
Panel_com('set_velfunc_id',[0 0]);pause(0.01);
Panel_com('send_gain_bias',[60.0,0,0,0]);pause(0.005);
Panel_com('set_position',[2 1]);pause(0.005);
Panel_com('start');
pause(2); 
%for j = 1:50
%    j
%    Panel_com('set_position',[1 2]);pause(0.005)
%    Panel_com('set_posfunc_id',[1 1]);pause(0.01);
%    Panel_com('set_funcx_freq', j*10);pause(0.1);
%   Panel_com('send_gai n_bias',[0,10,0,0]);pause(0.005);
%   Panel_com('set_mode',[4 0]);pause(0.05);
%    Panel_com('start');
%    pause(2);
%    Panel_com('stop');
%end;