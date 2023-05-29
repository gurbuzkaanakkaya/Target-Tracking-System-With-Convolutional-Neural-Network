#include <Servo.h>
#define MotorR1 8
#define MotorR2 9
#define MotorRE 5  // Motor pinlerini tanımlıyoruz.
#define MotorL1 10
#define MotorL2 11
#define MotorLE 6
int data_x1 = 0;
int data[2];
Servo myservo_x1; 
Servo myservo_y1;

void setup() {
  Serial.begin(9600);
  myservo_x1.attach(4);
  myservo_y1.attach(3);
  myservo_x1.write(90);
  myservo_y1.write(90);
  pinMode(MotorL1, OUTPUT);
  pinMode(MotorL2, OUTPUT);
  pinMode(MotorLE, OUTPUT); //Motorlarımızı çıkış olarak tanımlıyoruz.
  pinMode(MotorR1, OUTPUT);
  pinMode(MotorR2, OUTPUT);
  pinMode(MotorRE, OUTPUT);
}

void loop() {      
 while (Serial.available() >= 3) {       
     for (int i = 0; i < 3; i++) {
      data[i] = Serial.read();
     }
      myservo_x1.write(data[0]);
      myservo_y1.write(data[1]);
      Serial.println(data[0]);
      Serial.println(data[1]);
      Serial.println(data[2]);
     
  
     if (data[2]> 91) {  
         sag();                           
  }
  
     else if (data[2]<= 89 ) {
        sol();
        }
        
     else 
        sabit();
 }
}

void sag(){  // Robotun ileri yönde hareketi için fonksiyon tanımlıyoruz.

  digitalWrite(MotorR1, HIGH); // Sağ motorun ileri hareketi aktif
  digitalWrite(MotorR2, LOW); // Sağ motorun geri hareketi pasif
  analogWrite(MotorRE,150); // Sağ motorun hızı 150

  digitalWrite(MotorL1, LOW); // Sol motorun ileri hareketi aktif
  digitalWrite(MotorL2, LOW); // Sol motorun geri hareketi pasif
  analogWrite(MotorLE, 90); // Sol motorun hızı 150
}

void ileri(){  // Robotun ileri yönde hareketi için fonksiyon tanımlıyoruz.

  digitalWrite(MotorR1, LOW); // Sağ motorun ileri hareketi aktif
  digitalWrite(MotorR2, HIGH); // Sağ motorun geri hareketi pasif
  analogWrite(MotorRE, 70); // Sağ motorun hızı 150

  digitalWrite(MotorL1, HIGH); // Sol motorun ileri hareketi aktif
  digitalWrite(MotorL2, LOW); // Sol motorun geri hareketi pasif
  analogWrite(MotorLE, 70); // Sol motorun hızı 150
}

void sabit(){  // Robotun ileri yönde hareketi için fonksiyon tanımlıyoruz.

  digitalWrite(MotorR1, LOW); // Sağ motorun ileri hareketi aktif
  digitalWrite(MotorR2, LOW); // Sağ motorun geri hareketi pasif
  analogWrite(MotorRE, 0); // Sağ motorun hızı 150

  digitalWrite(MotorL1, LOW); // Sol motorun ileri hareketi aktif
  digitalWrite(MotorL2, LOW); // Sol motorun geri hareketi pasif
  analogWrite(MotorLE, 0); // Sol motorun hızı 150  
}

void sol(){ // Robotun geri yönde hareketi için fonksiyon tanımlıyoruz.

  digitalWrite(MotorR1, LOW); // Sağ motorun ileri hareketi pasif
  digitalWrite(MotorR2, LOW); // Sağ motorun geri hareketi aktif
  analogWrite(MotorRE,40); // Sağ motorun hızı 150

  digitalWrite(MotorL1, HIGH); // Sol motorun ileri hareketi pasif
  digitalWrite(MotorL2, LOW); // Sol motorun geri hareketi aktif
  analogWrite(MotorLE, 40); // Sol motorun hızı 150 
}
