# app.py
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable if frontend is separate

# MongoDB config
app.config["MONGO_URI"] = "mongodb://localhost:27017/radvision"
mongo = PyMongo(app)
users_collection = mongo.db.users


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password are required."}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists. Please log in."}), 400

    hashed_password = generate_password_hash(password)
    users_collection.insert_one({"email": email, "password": hashed_password})
    return jsonify({"message": "User registered successfully! Please log in."}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password are required."}), 400

    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        # You can return more user info here if needed
        return jsonify({
            "message": "Login successful!",
            "email": user["email"]
        }), 200
    elif user:
        return jsonify({"message": "Incorrect password."}), 401
    else:
        return jsonify({"message": "User not found. Please sign up."}), 404

if __name__ == "__main__":
    app.run(debug=True)
