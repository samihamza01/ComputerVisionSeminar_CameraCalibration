import serial

ser = serial.Serial("COM4",9600)

while 1:
	s = input("Enter something")
	sBytes = str.encode(s)
	ser.write(s)

	while ser.inWaiting() < 1:
		pass

	rec = ser.read(1)
	print(f"Echo: {rec.decode()}")