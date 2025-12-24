from flask import Flask, render_template, request
import brain
import rectina

app = Flask(__name__)

# ================= HOME PAGE =================
@app.route('/')
def home():
    return render_template('home.html')

# ================= BRAIN =====================
@app.route('/brain', methods=['GET', 'POST'])
def brain_route():
    if request.method == 'POST':
        return brain.brain_predict(request)
    return render_template('brain_index.html')

# ================= RETINA ====================
@app.route('/eye', methods=['GET', 'POST'])
def eye_route():
    if request.method == 'POST':
        return rectina.rectina_predict(request)
    return render_template('rectina_index.html')

if __name__ == "__main__":
    app.run(debug=True)
