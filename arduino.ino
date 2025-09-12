int segp[7] = {8, 9, 10, 11, 12, 13, A0};
const uint8_t digits[10][7] = {
  {1,1,1,1,1,1,0}, 
  {0,1,1,0,0,0,0}, 
  {1,1,0,1,1,0,1}, 
  {1,1,1,1,0,0,1}, 
  {0,1,1,0,0,1,1}, 
  {1,0,1,1,0,1,1}, 
  {1,0,1,1,1,1,1}, 
  {1,1,1,0,0,0,0}, 
  {1,1,1,1,1,1,1}, 
  {1,1,1,1,0,1,1}  
};
void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 3; i++) pinMode(i+2, OUTPUT);
  for (int i = 0; i < 7; i++) pinMode(segp[i], OUTPUT);
}
void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n'); 
    int l1, l2, l3, countdown = 0;
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      String ledPart = data.substring(0, commaIndex);
      String countPart = data.substring(commaIndex + 1);
      sscanf(ledPart.c_str(), "%d %d %d", &l1, &l2, &l3);
      countdown = countPart.toInt();
    } else {
      sscanf(data.c_str(), "%d %d %d", &l1, &l2, &l3);
    }
    digitalWrite(2, l1 ? HIGH : LOW);
    digitalWrite(3, l2 ? HIGH : LOW);
    digitalWrite(4, l3 ? HIGH : LOW);
    if (l1 == 1 && countdown > 0) {
      showDigit(countdown % 10);
    } else {
      showDigit(0); 
    }
  }
}
void showDigit(int num) {
  if (num < 0) num = 0;
  if (num > 9) num = num % 10;
  for (int i = 0; i < 7; i++) {
    digitalWrite(segp[i], digits[num][i] ? HIGH : LOW);  
  }
}