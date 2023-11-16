import cv2 as cv
import serial
import serial.tools.list_ports
import struct
from time import sleep
msgSet = {"REC": 1, "ACK": 2, "TRANSMISSION_OKAY": 3, "TRANSMISSION_FAIL": 4}

ports = serial.tools.list_ports.comports()

print("List of available devices:\n")
for port, desc, hwid in sorted(ports):
        print(f"\t{port}: {desc} [{hwid}]")

ser = serial.Serial(1)
sleep(5)


# Send the record image command
msg = struct.pack("B",msgSet["REC"])
ser.write(msg)

# Wait for information struct
while ser.inWaiting() < 16:
	pass

rxBytes = ser.read(16)
(width, height, format, length) = struct.unpack("<LLiL",rxBytes)

# Send ACK message
msg = struct.pack("<B",msgSet["ACK"])
ser.write(msg)

# recieve image in packages of 64 bytes, so the recieve buffer will never overflow
imageBytes = b''
for i in range(0,length/64):
	while ser.inWaiting() < 64:
		pass
	imageBytes = imageBytes + ',' + ser.read(64)

	# Send ACK message
	msg = struct.pack("<B",msgSet["ACK"])
	ser.write(msg)

while ser.inWaiting() < length%64:
	pass
imageBytes = imageBytes + ',' + ser.read(64)

# Send ACK message
msg = struct.pack("<B",msgSet["ACK"])
ser.write(msg)

ser.close()
with open("test.jpeg","wb") as binaryFile:
	binaryFile.write(imageBytes)