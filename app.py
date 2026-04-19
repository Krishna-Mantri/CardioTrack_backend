from flask import Flask, jsonify, request, send_file
from cardiolink import run_cardiolink_pipeline
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "CardioLink Flask API is running!"


@app.route('/generate-report', methods=['GET'])
def generate_report():
    try:
        result = run_cardiolink_pipeline()

        if not result or not os.path.exists(result):
            return jsonify({"error": "PDF file not found"}), 404

        return send_file(
            result,
            as_attachment=True,
            download_name="cardiolink_report.pdf",  # ✅ better UX
            mimetype='application/pdf'              # ✅ explicit type
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ IMPORTANT: Render compatibility
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render assigns PORT
    app.run(host='0.0.0.0', port=port)