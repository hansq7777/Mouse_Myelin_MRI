set terminal x11 size 780, 480 font ",20" title "Vfl Schedule Monitor"
x1 = system("cat schedules.dat | grep TE_AT_ECHO | cut -f 2 -d '='") + 0
y1 = system("cat schedules.dat | grep PL_HEIGHT | cut -f 2 -d '='")  + 0
T1  = system("cat schedules.dat | grep T1   | cut -f 2 -d '=' ") 
T2  = system("cat schedules.dat | grep T2   | cut -f 2 -d '=' ") 
TE  = system("cat schedules.dat | grep TE   | cut -f 2 -d '=' ") + 0
ESP = system("cat schedules.dat | grep ESP  | cut -f 2 -d '=' ") + 0
PL  = system("cat schedules.dat | grep PL_LENGTH | cut -f 2 -d '=' ") +0
PH  = system("cat schedules.dat | grep PL_HEIGHT | cut -f 2 -d '=' ") +0
set multiplot title sprintf("TE=%.2f,   ESP=%.2f,   T1=%s,   T2=%s\
\nPL: Length=%d,   Height=%.2f", TE, ESP, T1, T2, PL, PH)
set size 0.5, 0.85
set origin 0.5, 0.0
set xlabel "echo number"
set ylabel "flip angle"
set yrange [0:185]
p  "schedules.dat" u 1:2 w l lw 2 notitle
set origin 0.0, 0.0
set yrange [-0.01:1.05]
set ylabel "signal"
p "schedules.dat" u 1:3 w l lw 2 lc rgb "blue" notitle, \
 "schedules.dat" u 1:4 w lp lw 1 pt 7 ps 1 lc rgb "grey20" title "t2", \
  '+' u (x1):(y1) : (sprintf('te')) with labels offset char -1, 1\
  left textcolor rgb 'blue' point ls 5 lc rgb "red" notitle
unset multiplot
