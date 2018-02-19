"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
import time
import csv


TIME_DELAY = 0.0125
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
f = open('left.csv', 'w')
counter = 0
f.write(str([1,2,3,4,5,6,7,8])+'\n')
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
	sample, timestamp = inlet.pull_sample()
	if counter > 100:
		f.write(str(sample)+'\n')
		print(timestamp, sample)
	time.sleep(TIME_DELAY)
	if counter%500 == 0:
		print('\n\n\n\n','counter = ',counter,'\n\n\n\n')
	counter = counter + 1
	if counter >= 10000:
		break
		
f.close()
