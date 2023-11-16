#include "esp_camera.h"
#include <Arduino.h>

// for ESP logging
static const char* TAG = "main.cpp";

// Select camera model
//#define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_V2_PSRAM // M5Camera version B Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_ESP32CAM // No PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM

/* Defines 				*/

#define PWDN_GPIO_NUM   	32
#define RESET_GPIO_NUM  	-1
#define XCLK_GPIO_NUM   	 0
#define SIOD_GPIO_NUM   	26
#define SIOC_GPIO_NUM   	27
#define Y9_GPIO_NUM     	35
#define Y8_GPIO_NUM     	34
#define Y7_GPIO_NUM     	39
#define Y6_GPIO_NUM     	36
#define Y5_GPIO_NUM     	21
#define Y4_GPIO_NUM     	19
#define Y3_GPIO_NUM     	18
#define Y2_GPIO_NUM     	 5
#define VSYNC_GPIO_NUM  	25
#define HREF_GPIO_NUM   	23
#define PCLK_GPIO_NUM   	22

#define PACKAGE_SIZE_BYTES	128
#define CMD_ACK_SIZE_BYTES	1

							
#define REC 				1					// Record image cmd
#define ACK 				2					// Acknowlegement that package was received
#define TRANSMISSION_ERROR 	3					// Image transmission failed
#define TRANSMISSION_OKAY 	4					// Image transmission okay





/* Typedefs				*/ 
typedef struct {								// Image information struct
	long unsigned int width;					// Width of image in pixel
	long unsigned int height;					// Height of image in pixel
	pixformat_t format;							// Data Formate e.g. JPEG
	long unsigned int len;						// Length of the image
} image_information_t;

typedef union {									// Wrapper for image_information struct
	image_information_t info;					// image_information struct
	uint8_t raw[sizeof(image_information_t)];	// raw bytes
} image_information_tx_wrapper_t;


/* Global Variables 	*/
camera_config_t camera_config; 					// Configuration struct for cam
uint8_t uiTxBuffer[PACKAGE_SIZE_BYTES] = {0};					// TxBuffer for sending images
uint8_t uiRxBuffer[1] = {0};					// RxBuffer for recieving commands/ack
image_information_tx_wrapper_t image_info_tx;	// image information

/* Function prototypes 	*/
esp_err_t camera_init();
esp_err_t camera_capture(bool flashLight);
int process_image(long unsigned int width, long unsigned int height, pixformat_t format, uint8_t * buf, long unsigned int len);

void setup() {
	// Configure flash led
	pinMode(4,OUTPUT);

	// Set up serial communication
	Serial.begin(115200);

	// Init camera
	camera_init();
	
}

void loop() {
	if (Serial.available() >= 1) {
		Serial.read(uiRxBuffer, 1);
		if (uiRxBuffer[0] == REC) {
			camera_capture(true);
		}
	}
	
}

esp_err_t camera_init(){
    //power up the camera if PWDN pin is defined
    if(PWDN_GPIO_NUM != -1){
        pinMode(PWDN_GPIO_NUM, OUTPUT);
        digitalWrite(PWDN_GPIO_NUM, LOW);
    }
	camera_config.pin_pwdn = PWDN_GPIO_NUM;
	camera_config.pin_reset = RESET_GPIO_NUM;
	camera_config.pin_xclk = XCLK_GPIO_NUM;
	camera_config.pin_sccb_sda = SIOD_GPIO_NUM;
	camera_config.pin_sccb_scl = SIOC_GPIO_NUM;
	camera_config.pin_d7 = Y9_GPIO_NUM;
	camera_config.pin_d6 = Y8_GPIO_NUM;
	camera_config.pin_d5 = Y7_GPIO_NUM;
	camera_config.pin_d4 = Y6_GPIO_NUM;
	camera_config.pin_d3 = Y5_GPIO_NUM;
	camera_config.pin_d2 = Y4_GPIO_NUM;
	camera_config.pin_d1 = Y3_GPIO_NUM;
	camera_config.pin_d0 = Y2_GPIO_NUM;
	camera_config.pin_vsync = VSYNC_GPIO_NUM;
	camera_config.pin_href = HREF_GPIO_NUM;
	camera_config.pin_pclk = PCLK_GPIO_NUM;
	camera_config.xclk_freq_hz = 20000000;
	camera_config.ledc_timer = LEDC_TIMER_0;
	camera_config.ledc_channel = LEDC_CHANNEL_0;
	camera_config.pixel_format = PIXFORMAT_JPEG;

	// if PSRAM IC present, init with UXGA resolution and higher JPEG quality
	//                      for larger pre-allocated frame buffer.
	if(psramFound()){
		camera_config.frame_size = FRAMESIZE_UXGA;
		camera_config.jpeg_quality = 10;
		camera_config.fb_count = 2;
	} else {
		camera_config.frame_size = FRAMESIZE_SVGA;
		camera_config.jpeg_quality = 12;
		camera_config.fb_count = 1;
	}


    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera Init Failed");
		
        return err;
    }

	// Change some sensor settings
	sensor_t * s = esp_camera_sensor_get();
	s->set_brightness(s, 0);     // -2 to 2
	s->set_contrast(s, 0);       // -2 to 2
	s->set_saturation(s, 0);     // -2 to 2
	s->set_special_effect(s, 0); // 0 to 6 (0 - No Effect, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia)
	s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
	s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
	s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
	s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
	s->set_aec2(s, 0);           // 0 = disable , 1 = enable
	s->set_ae_level(s, 0);       // -2 to 2
	s->set_aec_value(s, 300);    // 0 to 1200
	s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
	s->set_agc_gain(s, 0);       // 0 to 30
	s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
	s->set_bpc(s, 0);            // 0 = disable , 1 = enable
	s->set_wpc(s, 1);            // 0 = disable , 1 = enable
	s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
	s->set_lenc(s, 1);           // 0 = disable , 1 = enable
	s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
	s->set_vflip(s, 0);          // 0 = disable , 1 = enable
	s->set_dcw(s, 1);            // 0 = disable , 1 = enable
	s->set_colorbar(s, 0);       // 0 = disable , 1 = enable

	// initial sensors are flipped vertically and colors are a bit saturated
	if (s->id.PID == OV3660_PID) {
		s->set_vflip(s, 1); // flip it back
		s->set_brightness(s, 1); // up the brightness just a bit
		s->set_saturation(s, -2); // lower the saturation
	}
	// drop down frame size for higher initial frame rate
	//s->set_framesize(s, FRAMESIZE_QVGA);

	// Initially capture some frames
	for (uint8_t i = 0; i < 10; i++) {
		camera_fb_t * fb = esp_camera_fb_get();
    	esp_camera_fb_return(fb);
    	fb = nullptr;
	}

    return ESP_OK;
}

esp_err_t camera_capture(bool flashLight) {
	// Frame buffer
	camera_fb_t * fb = NULL;

    // Acquire a frame
	if (flashLight) {
    	digitalWrite(4,HIGH);
    	fb = esp_camera_fb_get();
		digitalWrite(4,LOW);
	} else {
		fb = esp_camera_fb_get();
	}
    if (!fb) {
        ESP_LOGE(TAG, "Camera Capture Failed");
        return ESP_FAIL;
    }
	
    process_image(fb->width, fb->height, fb->format, fb->buf, fb->len);
  
    //return the frame buffer back to the driver for reuse
    esp_camera_fb_return(fb);
    return ESP_OK;
}

esp_err_t process_image(long unsigned int width, long unsigned int height, pixformat_t format, uint8_t * buf, long unsigned int len) {
	// fill in image information
	image_info_tx.info = {	.width = width,
							.height = height,
							.format = format,
							.len = len};


    // Note: ESP32Cam MB (Serial Interface/ Programmer) has no connections to enable HW Flow Control
	// Serial Transmission
	// Transfer image information
	Serial.write(image_info_tx.raw,sizeof(image_information_tx_wrapper_t));

	// wait for ack from receiver
	while (Serial.available() < 1);
	Serial.readBytes(uiRxBuffer,1);
	if (uiRxBuffer[0] != ACK) {
		ESP_LOGE(TAG, "Camera Capture Failed");
		return ESP_FAIL;
	}
	
	// Transfer data in PACKAGE_SIZE_BYTES byte packages
    for (unsigned long int i = 0; i < (unsigned long int)len/PACKAGE_SIZE_BYTES; i++) {
        Serial.write(buf+i*PACKAGE_SIZE_BYTES, PACKAGE_SIZE_BYTES);
		// Wait for ACK
		while (Serial.available() < 1);
		Serial.readBytes(uiRxBuffer,1);
		/// TODO: Check message 
    }
	int r = len%PACKAGE_SIZE_BYTES;
	Serial.write(buf+(len-(r+1)),r);
	// Wait for ACK
	while (Serial.available() < 1);
	Serial.readBytes(uiRxBuffer,1);
	/// TODO: Check message 

    return ESP_OK;
}