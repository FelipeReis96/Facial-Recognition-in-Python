import RPi.GPIO as GPIO
import time

# Define o pino GPIO ao qual o servo MG90 está conectado
SERVO_PIN = 18  # Altere conforme necessário

# Configurações do GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Configuração do PWM
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz (Período de 20ms)
pwm.start(0)

def set_angle(angle):
    """Define o ângulo do servo."""
    duty_cycle = (angle / 18.0) + 2  # Conversão de ângulo para ciclo de trabalho
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Tempo para o servo alcançar a posição
    pwm.ChangeDutyCycle(0)  # Evita jitter

try:
    while True:
        angle = int(input("Digite o ângulo (0 a 180): "))
        if 0 <= angle <= 180:
            set_angle(angle)
        else:
            print("Ângulo fora do intervalo! Insira um valor entre 0 e 180.")
except KeyboardInterrupt:
    print("\nEncerrando...")
    pwm.stop()
    GPIO.cleanup()
