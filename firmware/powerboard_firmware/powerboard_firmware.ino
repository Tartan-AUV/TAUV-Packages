size_t byteWritten = 0; // for incoming serial data
int voltage_pin = A0;
uint8_t buf[11];

typedef enum { 
  TEST_CODE = 0x01,
  HEARTBEAT = 0xFF,
  SET_POWER = 0x88,
  THRUSTER_POWER = 0x44,
  THRUSTER_POWER_V = 0x45,
  GET_VOLTAGE = 0x22
} CommandType;

void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}

struct RequestType {
  uint8_t command_id;
  union {
    //payload
  };
  uint16_t sumHash; 
};

struct ResponseType {
  uint8_t command_id;
  union {
    int voltage;
  };
  uint16_t sumHash;
};

void loop() {
  // send data only when you receive data:
  if (Serial.available() > 0) {
    // read the incoming byte:
    // struct RequestType testing;
    //uint8_t encoded_request[11];
    // memcpy(encoded_request, &GET_VOLTAGE, sizeof(CommandType));
    // memcpy(encoded_request[9], &hash(encoded_request), sizeof(uint16_t));

    // testing = parse(encoded_request);
    // Serial.println(testing.command_id, DEC);

    // uint8_t *encoded_response = createResponse(testing);

    // encoded_request[0] = GET_VOLTAGE;
    
    //byteWritten = Serial.readBytes(buf, 11);
    //uint16_t test = hash(all);

    // say what you got:
    //Serial.print("I received: ");
    //Serial.println(byteWritten, DEC);
    //Serial.println((char *)all);
    //printBytes(all, 11);
    //Serial.println(test, DEC);

    byteWritten = Serial.read();
    
    memcpy(buf, &buf[1], 10 * sizeof(uint8_t));
    buf[10] = (uint8_t) byteWritten;

    if (checkSum(buf)) {
      struct RequestType *parsedRequest = parse(buf);
      uint8_t *response = createResponse(*parsedRequest);

      Serial.write(response, 11);
    }
  }
}

// hash function
// @param buf 9 bytes char array with encoded message from jetson
// @return 2 bytes computed from rabin fingerprint
uint16_t hash(uint8_t buf[9]) {
  uint32_t checksum = 0;
  uint32_t powfactor = 1;
  for (int i = 0; i < 9; i++) {
    // checksum += ((uint16_t)((buf[i]) * pow(256, 9-1-i)) % 65521);
    checksum = (checksum + (buf[9-1-i] * powfactor)) % 65521;
    Serial.println(checksum, DEC);
    powfactor = (powfactor * 256) % 65521;
  }
  return checksum;
}

bool checkSum(uint8_t buf[11]) {
  return hash(buf) == *((uint16_t *)(&buf[9]));
}

struct RequestType *parse(uint8_t buf[11]) {
  uint8_t buffer[9];
  uint8_t response[11];
  memcpy(buffer, buf, sizeof(uint8_t) * 9);

  uint16_t sum = hash(buffer);
  uint16_t checksum;
  memcpy(&checksum, buf+9, sizeof(uint16_t));

  if (sum != checksum) {
    Serial.println(sum, DEC);
    Serial.println(checksum, DEC);
    return NULL;
  }

  struct RequestType request;

  switch (buf[0]) {
    case GET_VOLTAGE: {
      request.command_id = GET_VOLTAGE;
    }
  }

  return &request;
}

uint8_t *createResponse(struct RequestType request) {
  struct ResponseType response;
  uint8_t response_array[11];
  uint16_t checksum;

  switch (request.command_id) {
    case GET_VOLTAGE: {
      response.command_id = GET_VOLTAGE;
      response.voltage = analogRead(voltage_pin);

      memcpy(response_array, &response,9);
      checksum = hash(response_array);
      memcpy(response_array + 9, &checksum, sizeof(uint16_t));
    }
  }

  return response_array;
}

void printBytes(char *str, size_t length) {
  for (int i = 0; i < length; i++) {
    Serial.print(str[i], DEC);
  }
  Serial.println();
}

