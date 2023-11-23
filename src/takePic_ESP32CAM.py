import serial
import serial.tools.list_ports
import struct
from time import sleep
import io
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ESPCam():
	# attibutes
	port = ""
	baudrate = 115200
	PACKAGE_SIZE_BYTES = 128
	msgSet = {"REC": 1, "ACK": 2, "TRANSMISSION_OKAY": 3, "TRANSMISSION_FAIL": 4}
	binImageBuffer = b''
	ser = None

	# ctor
	def __init__(self,port) -> None:
		self.port = port
		return

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.ser.close()
		return

	def connect(self):
		self.ser = serial.Serial(self.port,115200)
		# Trigger reset
		self.ser.setDTR(True)
		self.ser.setDTR(False)
		self.ser.setRTS(False)
		sleep(0.1)
		self.ser.flushInput()
		self.ser.flushOutput()
		sleep(1)
		return
	
	def recordImage(self):
		# Send the record image command
		msg = struct.pack("B",self.msgSet["REC"])
		self.ser.write(msg)

		# Wait for information struct
		while self.ser.inWaiting() < 16:
			pass

		rxBytes = self.ser.read(16)
		(width, height, format, length) = struct.unpack("<LLiL",rxBytes)

		# Send ACK message
		msg = struct.pack("<B",self.msgSet["ACK"])
		self.ser.write(msg)

		# recieve image in packages of PACKAGE_SIZE_BYTES bytes, so the recieve buffer will never overflow
		self.binImageBuffer = b''
		for i in range(0,int(length/self.PACKAGE_SIZE_BYTES)):
			while self.ser.inWaiting() < self.PACKAGE_SIZE_BYTES:
				pass
			self.binImageBuffer = self.binImageBuffer + self.ser.read(self.PACKAGE_SIZE_BYTES)

			# Send ACK message
			msg = struct.pack("<B",self.msgSet["ACK"])
			self.ser.write(msg)

		while self.ser.inWaiting() < length%self.PACKAGE_SIZE_BYTES:
			pass
		self.binImageBuffer = self.binImageBuffer + self.ser.read(length%self.PACKAGE_SIZE_BYTES)

		# Send ACK message
		msg = struct.pack("<B",self.msgSet["ACK"])
		self.ser.write(msg)
		return

	def displayImageBuffer(self):
		fp = io.BytesIO(bytearray(self.binImageBuffer))
		with fp:
			img = Image.open(fp=fp)
			img.show()

	def storeImage(self,path):
		with open(path,"wb") as binaryFile:
			binaryFile.write(self.binImageBuffer)

	def disconnect(self):
		self.ser.close()

if __name__ == "__main__":
	try:
		cam = ESPCam("COM4")
		cam.connect()
		while 1:
			input("Hit enter to take picture: ")
			print("Processing...")
			cam.recordImage()
			print("Image received")
			cam.displayImageBuffer()
			cmd = input("Store the shown image? [y,n]")
			if cmd == "y" or cmd == "Y":
				filename = input("Enter filename for image: ")
				cam.storeImage("./calibration_images/"+filename+".jpeg")
				print("Image stored in: ./calibration_images/"+filename+".jpeg")
			cmd = input("Record another image? [y,n]: ")
			if cmd == "n" or cmd == "N":
				break
		print("Disconnecting...")
		cam.disconnect()
		print("End of programm.")
	except:
		print("Error Occured: Disconnecting from Camera...")
		cam.disconnect()
		print("End Programm")