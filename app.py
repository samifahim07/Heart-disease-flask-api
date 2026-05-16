from flask import Flask, request, jsonify, render_template
import pickle, warnings
import pandas as pd

warnings.filterwarnings("ignore")
app = Flask(__name__, template_folder="template")
model = pickle.load(open("model.pkl", "rb"))

# 13 columns in the exact order the model was trained on
EXPECTED_COLS = [' ""', 'Age', 'Sex', 'ChestPain', 'RestBP', 'Chol',
                 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca','Thal']

# LabelEncoder mapping from training notebook cell 30 (alphabetical):
CP_MAP = {
    "asymptomatic": 0,"nonanginal": 1,"nontypical": 2,"typical": 3,"atypical": 2, }

NUMERIC_FIELDS = ["Age","Sex","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca"]

@app.route("/")
def home():
    return render_template("index.html")

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if request.method == "GET":
        return jsonify({
            "message": "Send a POST request with JSON body.",
            "required_fields": NUMERIC_FIELDS + ["ChestPain"],
            "ChestPain_values": ["asymptomatic","nonanginal","nontypical","typical"]
        })

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Empty or invalid JSON body"}), 400

        # ChestPain → integer (matches LabelEncoder used in training)
        cp_raw = str(data.get("ChestPain","")).lower().strip()
        if cp_raw not in CP_MAP:
            return jsonify({"error": f"Invalid ChestPain '{cp_raw}'. "
                                     f"Use: asymptomatic, nonanginal, nontypical, typical"}), 400
        cp_encoded = CP_MAP[cp_raw]

        
        missing = [f for f in NUMERIC_FIELDS if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        row = {
            ' ""': 0, 
            'Age': float(data["Age"]),
            'Sex': int(data["Sex"]),
            'ChestPain': cp_encoded, 
            'RestBP': float(data["RestBP"]),
            'Chol': float(data["Chol"]),
            'Fbs': int(data["Fbs"]),
            'RestECG': int(data["RestECG"]),
            'MaxHR': float(data["MaxHR"]),
            'ExAng': int(data["ExAng"]),
            'Oldpeak': float(data["Oldpeak"]),
            'Slope': int(data["Slope"]),
            'Ca': float(data["Ca"]),
            'Thal': int(data["Thal"]),
        }

       
        X = pd.DataFrame([row], columns=EXPECTED_COLS)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        confidence = round(float(max(probability)) * 100, 2)

        return jsonify({
            "prediction": int(prediction),
            "has_heart_disease": bool(prediction),
            "confidence": confidence,
            "probabilities": {
                "No": round(float(probability[0]) * 100, 2),
                "Yes": round(float(probability[1]) * 100, 2),
            },
            "echo": {**{k: row[k] for k in EXPECTED_COLS if k != ' ""'},
                     "ChestPain_decoded": cp_raw},
        })

    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)