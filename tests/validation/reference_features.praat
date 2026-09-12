form Reference metrics
 sentence audio_path /tmp/input.wav
 positive pitch_floor 75
 positive pitch_ceiling 500
endform
sound = Read from file: audio_path$
To PointProcess (periodic, cc): pitch_floor, pitch_ceiling
pp = selected("PointProcess")
jitter = Get jitter (local): 0, 0, 0.0001, 0.02, 1.3
selectObject: sound, pp
shimmer = Get shimmer (local): 0, 0, 0.0001, 0.02, 1.3, 1.6
selectObject: sound
To Harmonicity (cc): 0.01, 75, 0.1, 1
hnr = Get mean: 0, 0
selectObject: sound
To Intensity: 100, 0.01, "yes"
n = Get number of frames
sum = 0
count = 0
for i from 1 to n
 value = Get value in frame: i
 if value > 0
  sum = sum + value
  count = count + 1
 endif
endfor
intmean = sum / count
selectObject: sound
To Spectrum: "yes"
cog = Get centre of gravity: 2
lo500 = Get band energy: 0, 500
nyquist = Get highest frequency
hi500 = Get band energy: 500, nyquist
lo1000 = Get band energy: 0, 1000
hi1000 = Get band energy: 1000, nyquist
ratio500 = hi500 / lo500
ratio1000 = hi1000 / lo1000
writeInfoLine: "Jitter,Shimmer,HNR_dB,Int_mean,COG_Hz,HF500_ratio,HF1000_ratio"
appendInfoLine: jitter, ",", shimmer, ",", hnr, ",", intmean, ",", cog, ",", ratio500, ",", ratio1000
