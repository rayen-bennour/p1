from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from any origin. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model("student_performance_model")

@app.get("/", response_class=HTMLResponse)
async def root():
    # Read and return the HTML file
    with open("static/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content, status_code=200)

@app.get("/predict")
async def predict(
    Age: int = Query(..., description="Student Age"),
    Gender: int = Query(..., description="Gender"),
    HighSchoolType: int = Query(..., description="High School Type"),
    Scholarship: int = Query(..., description="Scholarship Type"),
    AdditionalWork: int = Query(..., description="Additional Work"),
    SportsActivity: int = Query(..., description="Regular Artistic or Sports Activity"),
    Partner: int = Query(..., description="Do you have a partner"),
    Salary: int = Query(..., description="Total Salary if Available"),
    Transportation: int = Query(..., description="Transportation to the University"),
    Accommodation: int = Query(..., description="Accommodation Type in Cyprus"),
    MotherEducation: int = Query(..., description="Mother’s Education"),
    FatherEducation: int = Query(..., description="Father’s Education"),
    Siblings: int = Query(..., description="Number of Siblings"),
    ParentalStatus: int = Query(..., description="Parental Status"),
    MotherOccupation: int = Query(..., description="Mother’s Occupation"),
    FatherOccupation: int = Query(..., description="Father’s Occupation"),
    StudyHours: int = Query(..., description="Weekly Study Hours"),
    ReadingNonScientific: int = Query(..., description="Reading Frequency (Non-Scientific Books/Journals)"),
    ReadingScientific: int = Query(..., description="Reading Frequency (Scientific Books/Journals)"),
    SeminarAttendance: int = Query(..., description="Attendance to Seminars/Conferences")
):
    data = {
        "Student Age": Age,
        "Sex": Gender,
        "Graduated high-school type": HighSchoolType,
        "Scholarship type": Scholarship,
        "Additional work": AdditionalWork,
        "Regular artistic or sports activity": SportsActivity,
        "Do you have a partner": Partner,
        "Total salary if available": Salary,
        "Transportation to the university": Transportation,
        "Accommodation type in Cyprus": Accommodation,
        "Mothers’ education": MotherEducation,
        "Fathers’ education": FatherEducation,
        "Number of sisters/brothers": Siblings,
        "Parental status": ParentalStatus,
        "Mothers’ occupation": MotherOccupation,
        "Fathers’ occupation": FatherOccupation,
        "Weekly study hours": StudyHours,
        "Reading frequency (non-scientific books/journals)": ReadingNonScientific,
        "Reading frequency (scientific books/journals)": ReadingScientific,
        "Attendance to the seminars/conferences related to the department": SeminarAttendance
    }
    
    df = pd.DataFrame([data])
    predictions = predict_model(model, data=df)
    predicted_grade = predictions["prediction_label"].iloc[0]
    
    grade_map = {
        0: 'Excellent (90-100%)',
        1: 'Very Good (80-89%)',
        2: 'Good (70-79%)',
        3: 'Acceptable (60-69%)',
        4: 'Fail (Below 60%)'
    }
    grade = grade_map.get(predicted_grade, 'Unknown')

    return {"Predicted Grade": grade}
