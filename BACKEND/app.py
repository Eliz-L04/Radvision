# app.py
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
mongo = PyMongo(app)
users_collection = mongo.db.users

@app.route("/")
def home():
    return jsonify({"message": "✅ Flask server is running with MongoDB Atlas"})

@app.route("/test-db", methods=["GET"])
def test_db():
    try:
        collections = mongo.db.list_collection_names()
        return jsonify({"status": "success", "collections": collections}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    dob = data.get("dob")

    if not username or not email or not password or not dob:
        return jsonify({"message": "All fields are required."}), 400

    # Password validation: 8+ chars, at least 1 number
    if len(password) < 8 or not any(char.isdigit() for char in password):
        return jsonify({"message": "Password must be at least 8 characters and contain at least one number."}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists. Please log in."}), 400

    hashed_password = generate_password_hash(password)
    users_collection.insert_one({
        "username": username,
        "email": email,
        "password": hashed_password,
        "dob": dob
    })
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
        return jsonify({
            "message": "Login successful!",
            "email": user["email"],
            "username": user.get("username", "User")
        }), 200
    elif user:
        return jsonify({"message": "Incorrect password."}), 401
    else:
        return jsonify({"message": "User not found. Please sign up."}), 404

if __name__ == "__main__":
    app.run(debug=True)
