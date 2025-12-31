import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder="templates")

# ================= LOAD MODELS =================
model = pickle.load(open('model.pkl', 'rb'))     # placement model
model1 = pickle.load(open('model1.pkl', 'rb'))   # salary model


# ================= ROUTES =================
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    return render_template('index.html')


# ================= SUGGESTION FUNCTION =================
def get_suggestions(brr):
    suggestions = []

    cgpa, projects, workshops, mini_projects, skills, communication, internship, hackathon, tw, te, backlogs = brr

    if cgpa < 7:
        suggestions.append("📘 Improve your CGPA to at least 7.5")

    if projects < 2:
        suggestions.append("💻 Work on more academic / real‑world projects")

    if workshops < 2:
        suggestions.append("🛠 Attend more workshops or technical seminars")

    if mini_projects < 2:
        suggestions.append("📂 Add more mini‑projects to your resume")

    if skills < 5:
        suggestions.append("🧠 Learn more technical skills (DSA, Python, ML, Web)")

    if communication < 6:
        suggestions.append("🗣 Improve communication & interview skills")

    if internship == 0:
        suggestions.append("🏢 Try to complete at least one internship")

    if hackathon == 0:
        suggestions.append("🏆 Participate in hackathons or coding contests")

    if tw < 70 or te < 70:
        suggestions.append("📊 Improve academic percentages")

    if backlogs > 0:
        suggestions.append("❗ Clear active backlogs")

    if not suggestions:
        suggestions.append("✅ You already have a strong profile. Keep it up!")

    return suggestions


# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['GET'])
def predict():

    name = request.args.get('name', 'Student')

    # 1️⃣ Convert inputs safely
    try:
        cgpa = float(request.args.get('cgpa', 0))
        projects = int(request.args.get('projects', 0))
        workshops = int(request.args.get('workshops', 0))
        mini_projects = int(request.args.get('mini_projects', 0))
        skills_text = request.args.get('skills', '')
        communication = int(request.args.get('communication_skills', 0))
        internship = int(request.args.get('internship', 0))
        hackathon = int(request.args.get('hackathon', 0))
        tw = float(request.args.get('tw_percentage', 0))
        te = float(request.args.get('te_percentage', 0))
        backlogs = int(request.args.get('backlogs', 0))
    except ValueError:
        return "Please enter valid numeric values."

    # 2️⃣ Skill count
    s = len([i for i in skills_text.split(',') if i.strip()])

    # 3️⃣ List for suggestions (VERY IMPORTANT)
    brr_list = [
        cgpa, projects, workshops, mini_projects, s,
        communication, internship, hackathon,
        tw, te, backlogs
    ]

    # 4️⃣ Suggestions
    suggestions = get_suggestions(brr_list)
    print("SUGGESTIONS",suggestions)

    # 5️⃣ Model probability prediction
    brr = np.array(brr_list).reshape(1, -1)
    proba = model.predict_proba(brr)
    placed_probability = round(proba[0][1] * 100, 2)

    # 6️⃣ Placement status
    if placed_probability >= 50:
        p = 1
        placement_status = "Placed"
    else:
        p = 0
        placement_status = "Not Placed"

    # 7️⃣ Salary prediction
    salary_input = np.append(brr_list, p).reshape(1, -1)
    salary = model1.predict(salary_input)

    k = str(int(salary[0]))
    if len(k) == 6:
        k = k[0] + ',' + k[1:3] + ',' + k[3:]
    elif len(k) == 7:
        k = k[:2] + ',' + k[2:4] + ',' + k[4:]

    # 8️⃣ Output messages
    if placement_status == "Placed":
        out = f"Congratulations {name} 🎉<br>Placement Probability: {placed_probability}%"
        out2 = f"Your Expected Salary will be INR {k} per annum"
    else:
        out = f"Sorry {name}<br>Placement Probability: {placed_probability}%"
        out2 = "Improve your skills to increase your chances."

    # 9️⃣ Render page
    return render_template(
        'output.html',
        output=out,
        output2=out2,
        suggestions=suggestions
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)



