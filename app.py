from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load('naive_bayes_model.pkl')

# สร้างแอป Flask
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

# กำหนด endpoint สำหรับการทำนาย
@app.route('/predict', methods=['POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # รับข้อมูลจากฟอร์ม
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # ดึงข้อมูลที่จำเป็นจากฟอร์ม
            features = [N, P, K, temperature, humidity, ph, rainfall]

            # แปลงข้อมูลเป็น array
            features = np.array(features).reshape(1, -1)

            # ทำนายผล
            prediction = model.predict(features)

            # เก็บผลลัพธ์ที่ทำนายได้
            result = prediction[0]

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)
# เริ่มเซิร์ฟเวอร์
if __name__ == '__main__':
    app.run(debug=True)
